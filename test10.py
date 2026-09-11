import sys
import logging
from pyspark.context import SparkContext
from pyspark.sql.functions import col, to_date, date_format
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("warehouse_loader")

# Shared date pattern constants - use lowercase 'yyyy' (calendar year), NOT
# uppercase 'YYYY' (ISO week-based year, which requires an accompanying 'w'
# field and is rejected by Spark 3.x's strict/CORRECTED date parser).
RAW_DATE_PATTERN = "yyyy/MM/dd"
FISCAL_YEAR_PATTERN = "yyyy"

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Make the date-time parser policy explicit so behaviour does not silently
# drift across Spark/Glue version upgrades.
spark.conf.set("spark.sql.legacy.timeParserPolicy", "CORRECTED")

logger.info("Reading stage supply-chain invoice log entries...")

invoice_records = [
    ("INV-5001", "2026/01/15", 4200.00),
    ("INV-5002", "2026/02/20", 1850.50),
    ("INV-5003", "2026/03/10", 940.25)
]
invoice_df = spark.createDataFrame(invoice_records, ["invoice_id", "date_received", "amount_billed"])

logger.info("Normalizing date types into ISO partition-compliant schema...")

try:
    # Parse the raw date string into a proper DateType column using the
    # calendar-year pattern 'yyyy/MM/dd' (lowercase 'yyyy'), then derive
    # fiscal_year from the already-normalized DateType column rather than
    # re-parsing the original string a second time.
    clean_invoices = invoice_df.withColumn(
        "normalized_date",
        to_date(col("date_received"), RAW_DATE_PATTERN)
    ).withColumn(
        "fiscal_year",
        date_format(col("normalized_date"), FISCAL_YEAR_PATTERN)
    )

    # Validate that parsing actually succeeded for every row. Any row where
    # normalized_date is null despite a non-null date_received indicates a
    # malformed date string that slipped past the pattern.
    bad_rows = clean_invoices.filter(
        col("date_received").isNotNull() & col("normalized_date").isNull()
    )
    bad_row_count = bad_rows.count()
    if bad_row_count > 0:
        bad_samples = [r["date_received"] for r in bad_rows.select("date_received").limit(5).collect()]
        logger.error(
            "Found %d invoice record(s) with unparseable date_received values. Samples: %s",
            bad_row_count, bad_samples
        )
        raise ValueError(
            "Date normalization produced null normalized_date for %d row(s); "
            "sample offending date_received values: %s" % (bad_row_count, bad_samples)
        )

    logger.info("Displaying standardized invoice batch...")
    clean_invoices.show()
except Exception:
    logger.error("Invoice date normalization failed. Schema of invoice_df for diagnostics:")
    try:
        invoice_df.printSchema()
    except Exception:
        logger.error("Unable to print invoice_df schema during error handling.")
    raise
finally:
    # Ensure Glue job bookkeeping/CloudWatch metrics are always reported,
    # even if the transformation stage above fails and re-raises.
    job.commit()
