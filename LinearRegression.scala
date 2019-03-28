// Import LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

// Optional: Use the following code below to set the Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Start a simple Spark Session
val spark = SparkSession.builder().appName("Linear Regression Scala Project").getOrCreate()

// Use Spark to read in the Ecommerce Customers csv file.
val data = spark.read.option("header","true").option("inferschema","true").format("csv").load("Clean-Ecommerce.csv")

// Print the Schema of the DataFrame
data.printSchema()

// Print out an example Row
// Various ways to do this, just
// choose whichever way you prefer
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
for (ind <- Range(1,colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}

// Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Rename the Yearly Amount Spent Column as "label"
// Also grab only the numerical columns from the data
// Set all of this as a new dataframe called df
val df = (data.select(data("Yearly Amount Spent").as("label"),$"Avg Session Length",
$"Time on App", $"Time on Website", $"Length of Membership"))

// Use VectorAssembler to convert the input columns of df
// to a single output column of an array called "features"
// Set the input columns from which we are supposed to read the values.
// Call this new object assembler
val assembler = (new VectorAssembler().setInputCols(Array("Avg Session Length",
"Time on App","Time on Website", "Length of Membership")).setOutputCol("features"))

// Use the assembler to transform our DataFrame to the two columns: label and features
val output = assembler.transform(df).select($"label",$"features")

// Create a Linear Regression Model object
val lr = new LinearRegression()

// Fit the model to the data and call this model lrModel
 val lrModel = lr.fit(output)

// Print the coefficients and intercept for linear regression
println(s"Intecept is ${lrModel.intercept}")
println(s"Intecept is ${lrModel.coefficients}")

// Summarize the model over the training set and print out some metrics!
// Use the .summary method off your model to create an object
// called trainingSummary
val trainingSummary = lrModel.summary

// Show the residuals, the RMSE, the MSE, and the R^2 Values.
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE : ${trainingSummary.meanSquaredError}")
println(s"MSE : ${trainingSummary.r2}")
