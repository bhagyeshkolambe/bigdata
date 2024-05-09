from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col

# Create a SparkSession
spark = SparkSession.builder \
    .appName("TextClustering") \
    .getOrCreate()

# Sample text data
data = [
    (1, "This is the first document."),
    (2, "This document is the second document."),
    (3, "And this is the third one."),
    (4, "Is this the first document?")
]

# Create a DataFrame
df = spark.createDataFrame(data, ["id", "text"])

# Tokenize the text
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(df)

# Convert words to TF
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

# Convert TF to TF-IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Apply KMeans clustering
num_clusters = 2  # You can set the number of clusters you want
kmeans = KMeans(k=num_clusters, seed=1)
model = kmeans.fit(rescaledData)

# Get cluster labels
predictions = model.transform(rescaledData)

# Print the clusters
predictions.select("id", "text", "prediction").show()

# Stop the SparkSession
spark.stop()
