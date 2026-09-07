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

# FIX: Replaced 'YYYY' (ISO 8601 week-based year) with 'yyyy' (proleptic Gregorian calendar year)
# Spark 3.x delegates to Java DateTimeFormatter which strictly rejects 'YYYY' without an
# accompanying week-of-year field ('ww'), raising SparkUpgradeException at action time.
# The correct token for a calendar year (e.g. 2026) is lowercase 'yyyy'.
clean_invoices = invoice_df.withColumn(
    "normalized_date",
    to_date(col("date_received"), "yyyy/MM/dd")
).withColumn("fiscal_year", date_format(col("date_received"), "yyyy"))

logger.info("Displaying standardized invoice batch...")
clean_invoices.show()
job.commit()
