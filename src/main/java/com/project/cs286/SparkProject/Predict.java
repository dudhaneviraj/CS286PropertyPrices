package com.project.cs286.SparkProject;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.TreeEnsembleModel;

import scala.Tuple2;

public class Predict {
	

	
	public static void predictAccuracy(JavaRDD<LabeledPoint> testData,LinearRegressionModel model)
	{
		JavaPairRDD<Double, Double> predictionAndLabel =testData.mapToPair(p->{
			return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
		});
	
		predictionAndLabel.saveAsTextFile("RESULT");
			
		Double trainMSE=predictionAndLabel.map(f->{
			Double diff =f._1()-f._2();  
			return Math.pow(diff, 2);})
				.reduce(
						(p1,p2)->{ return (p1+p2);})/testData.count();
		
		System.out.println("Training Mean Squared Error: " + trainMSE);
		System.out.println("Learned regression tree model:\n" + model);
	}
	
	public static void predictAccuracy(JavaRDD<LabeledPoint> testData,TreeEnsembleModel model)
	{
		
		JavaPairRDD<Double, Double> predictionAndLabel =testData.mapToPair(p->{
			return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
		});
		
		predictionAndLabel.saveAsTextFile("RESULT");
		
		
		Double trainMSE=predictionAndLabel.map(f->{
			Double diff =f._1()-f._2();  
			return Math.pow(diff, 2);})
				.reduce(
						(p1,p2)->{ return (p1+p2);})/testData.count();
		
		System.out.println("Training Mean Squared Error: " + trainMSE);
		System.out.println("Learned regression tree model:\n" + model);
	}

	public static void predictAccuracy(JavaRDD<LabeledPoint> testData,DecisionTreeModel model)
	{
		
		JavaPairRDD<Double, Double> predictionAndLabel =testData.mapToPair(p->{
			return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
		});
		
		predictionAndLabel.saveAsTextFile("RESULT");
		
		
		Double trainMSE=predictionAndLabel.map(f->{
			Double diff =f._1()-f._2();  
			return Math.pow(diff, 2);})
				.reduce(
						(p1,p2)->{ return (p1+p2);})/testData.count();
		
		System.out.println("Training Mean Squared Error: " + trainMSE);
		System.out.println("Learned regression tree model:\n" + model);
	}

}
