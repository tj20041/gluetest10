import sys
import logging
from pyspark.context import SparkContext
from pyspark.sql.functions import col, to_date, date_format
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("warehouse_loader")

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger.info("Reading stage supply-chain invoice log entries...")

invoice_records = [
    ("INV-5001", "2026/01/15", 4200.00),
    ("INV-5002", "2026/02/20", 1850.50),
    ("INV-5003", "2026/03/10", 940.25)
]
invoice_df = spark.createDataFrame(invoice_records, ["invoice_id", "date_received", "amount_billed"])

logger.info("Normalizing date types into ISO partition-compliant schema...")

# FIX: Replaced uppercase 'YYYY' (ISO week-based year, invalid in Spark 3.x without a week-of-year
# specifier) with lowercase 'yyyy' (proleptic Gregorian calendar year) in both to_date() and
# date_format() calls. Spark 3.x's strict DateTimeFormatter rejects 'YYYY' in a non-week context,
# raising SparkUpgradeException INCONSISTENT_BEHAVIOR_CROSS_VERSION.DATETIME_PATTERN_RECOGNITION.
clean_invoices = invoice_df.withColumn(
    "normalized_date",
    to_date(col("date_received"), "yyyy/MM/dd")
).withColumn("fiscal_year", date_format(col("date_received"), "yyyy"))

logger.info("Clean invoice count: %d", clean_invoices.count())
job.commit()
