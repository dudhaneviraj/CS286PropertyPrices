package com.project.cs286.SparkProject;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;

public class App {
	public static void spark() {
		SparkConf conf = new SparkConf().setMaster("local").setAppName("SPARK SQL Application");

		// Create a Java version of the Spark Context from the configuration
		JavaSparkContext context = new JavaSparkContext(conf);

		SQLContext sqlContext = new SQLContext(context);
		DataFrame dataFrame = sqlContext.read().format("com.databricks.spark.csv").option("inferSchema", "true")
				.option("header", "false").load("/home/viraj/temp.csv");
		dataFrame.show();

		StringIndexer indexer = new StringIndexer().setInputCol("C2").setOutputCol("PType");
		DataFrame pType = indexer.fit(dataFrame).transform(dataFrame);

		indexer = new StringIndexer().setInputCol("C3").setOutputCol("Status");
		DataFrame status = indexer.fit(pType).transform(pType);

		indexer = new StringIndexer().setInputCol("C5").setOutputCol("District");
		DataFrame district = indexer.fit(status).transform(status);

		indexer = new StringIndexer().setInputCol("C6").setOutputCol("County");
		DataFrame county = indexer.fit(district).transform(district);

		DataFrame data = county.drop("C2").drop("C3").drop("C4").drop("C5").drop("C6");

		data.printSchema();

//		JavaRDD<Row> row = data.toJavaRDD();
//
//		JavaRDD<LabeledPoint> parsedData = row.map(s -> {
//			return new LabeledPoint(Double.parseDouble(String.valueOf(s.getInt(0))),
//					Vectors.dense(Double.parseDouble(String.valueOf(s.getInt(1))), s.getDouble(2), s.getDouble(3),
//							s.getDouble(4), s.getDouble(5)));
//		});
//		
//		parsedData.cache();
//		JavaRDD.toRDD(parsedData).saveAsTextFile("output");
//
//		// Split initial RDD into two... [60% training data, 40% testing data].
//		JavaRDD<LabeledPoint>[] splits = parsedData.randomSplit(new double[] { 0.8, 0.2 });
//		JavaRDD<LabeledPoint> training = splits[0];
//		JavaRDD<LabeledPoint> testData = splits[1];
//	
//		
//		parsedData.saveAsTextFile("Training");
//		
//		SparkContext ctxt=JavaSparkContext.toSparkContext(context);
////		decisionTreeRegression(ctxt,training, testData);
////		linearRegression(ctxt,training, testData);
//		randomForestRegression(ctxt,parsedData, parsedData);

	}
	
	
	
	
	public static void randomForestRegression(SparkContext context,JavaRDD<LabeledPoint> training, JavaRDD<LabeledPoint> testData) {
		HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
		Integer numTrees = 4; // Use more in practice.
		String featureSubsetStrategy = "auto"; // Let the algorithm choose.
		String impurity = "variance";
		Integer maxDepth = 28;
		Integer maxBins = 500;
		Integer seed = 12345;

		RandomForestModel model = RandomForest.trainRegressor(training,
		  categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
		  seed);
		
		
		
		model.save(context,"RandomForest" );
		// Evaluate model on test instances and compute test error
		Predict.predictAccuracy(testData, model);
		
	}
	
	public static void linearRegression(SparkContext context,JavaRDD<LabeledPoint> training, JavaRDD<LabeledPoint> testData) {
		LinearRegressionModel model=LinearRegressionWithSGD.train(JavaRDD.toRDD(training), 1);
		model.save(context,"LinearRegression" );
		Predict.predictAccuracy(testData, model);
	}
	

	public static void decisionTreeRegression(SparkContext context,JavaRDD<LabeledPoint> training, JavaRDD<LabeledPoint> testData) {
		// Decison tree
		HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
		String impurity = "variance";
		Integer maxDepth = 30;
		Integer maxBins = 170;
		// Train a DecisionTree model for classification.
		
		DecisionTreeModel model = DecisionTree.trainRegressor(training, categoricalFeaturesInfo, impurity,
				maxDepth, maxBins);
		model.save(context,"DecisionTree");
		Predict.predictAccuracy(testData, model);
	}

	public static void main(String[] args) throws IOException {

		FileUtils.deleteDirectory(new File("output"));
		FileUtils.deleteDirectory(new File("RESULT"));
		FileUtils.deleteDirectory(new File("Training"));
		FileUtils.deleteDirectory(new File("DecisionTree"));
		FileUtils.deleteDirectory(new File("RandomForest"));
		FileUtils.deleteDirectory(new File("LinearRegression"));
		spark();
//		SparkConf conf = new SparkConf().setMaster("local").setAppName("SPARK SQL Application");
//
//		// Create a Java version of the Spark Context from the configuration
//		JavaSparkContext context = new JavaSparkContext(conf);
//
//		DecisionTreeModel modelD=DecisionTreeModel.load(context.sc(), "DecisionTree");
	
		
	}
}
