import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}

/**
 * Created by Tanmay on 7/10/2015.
 */
object LinearRegression {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "LinearRegression")

    val rawData = sc.textFile("D:\\NEU - Big Data and Intelligent Analytics\\Midterm\\YearPredictionMSD.txt")



    //converting data to LabeledPoint RDD
    val Data1 = rawData.map { line =>
      val parts = line.split(",")
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts.slice(1,89).map(_.toDouble)))
    }.cache()


    val vectors = Data1.map(lp => lp.features)

    val scaler = new StandardScaler(withMean = true, withStd =
      true).fit(vectors)
    val parsedData = Data1.map(lp => LabeledPoint(lp.label,
      scaler.transform(lp.features)))


    //splitting into training and test dataset
    val splits = parsedData.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)




    // Building the model
    val numIterations = 100
    val model = LinearRegressionWithSGD.train(training, numIterations,stepSize = 1)
/*
    // Evaluate model on training examples and compute training error
    val valuesAndPreds = training.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

*/
    //testing the model

    val valuesAndPreds1 = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }



    println(model.weights)


 //   val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
 //   println("training Root Mean Squared Error = " + math.sqrt(MSE))

    val MSE1 = valuesAndPreds1.map{case(v, p) => math.pow((v - p), 2)}.mean()
    println("test Root Mean Squared Error = " + math.sqrt(MSE1))



    sc.stop()

  }

}








