using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using SearchML.ML.Models;

namespace SearchML
{
    class Program
    {
        private const string DATA_FILEPATH = @"D:\Projects\C#\SearchML\test.csv";

        static void Main(string[] args)
        {
            // Create single instance of sample data from first line of dataset for model input
            SearchHistoryInput sampleData = CreateSingleDataSample(DATA_FILEPATH);
            
            
            
            // Create new MLContext
            MLContext mlContext = new MLContext();

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<SearchHistoryInput>(
                path: DATA_FILEPATH,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                allowSparse: false);
            
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(mlContext);
            
            IDataView transformedData = trainingPipeline.Fit(trainingDataView).Transform(trainingDataView);
            
            // Getting the resulting data as an IEnumerable.
            // This will contain the newly created columns.
            IEnumerable<SearchHistoryInputTransformed> features = mlContext.Data.CreateEnumerable<
                SearchHistoryInputTransformed>(transformedData, reuseRowObject: false);
            
            // write the result of transformed data
            foreach (var data in features)
            {
                Console.WriteLine($"source {data.Destination}:{data.Destination}");
            }
            
            ITransformer mlModel = TrainModel(mlContext, trainingDataView, trainingPipeline);
            
            // Load model & create prediction engine

            var predEngine = mlContext.Model.CreatePredictionEngine<SearchHistoryInput, PredictionOutput>(mlModel);

            // Use model to make prediction on input data
            PredictionOutput predictionResult = predEngine.Predict(sampleData);

            Console.WriteLine("Using model to make single prediction -- Comparing actual Destination with predicted Destination from sample data...\n\n");
            Console.WriteLine($"source: {sampleData.Source}");
            Console.WriteLine($"\n\nActual Destination: {sampleData.Destination} \nPredicted Destination value {predictionResult.Prediction} \nPredicted Destination scores: [{String.Join(",", predictionResult.Score)}]\n\n");
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }
        
        private static SearchHistoryInput CreateSingleDataSample(string dataFilePath)
        {
            // Create MLContext
            MLContext mlContext = new MLContext();

            // Load dataset
            IDataView dataView = mlContext.Data.LoadFromTextFile<SearchHistoryInput>(
                path: dataFilePath,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                allowSparse: false);

            // Use first line of dataset as model input
            // You can replace this with new test data (hardcoded or from end-user application)
            SearchHistoryInput sampleForPrediction = mlContext.Data.CreateEnumerable<SearchHistoryInput>(dataView, false)
                .First();
            return sampleForPrediction;
        }
        
        public static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(new []
                {
                    new InputOutputColumnPair("destinationCategory", "destination"),
                })
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("source", "source"))
                .Append(mlContext.Transforms.CopyColumns("Features", "source"))
                .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                .AppendCacheCheckpoint(mlContext);

            // Set the training algorithm 
            var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "destination", numberOfIterations: 10, featureColumnName: "Features"), labelColumnName: "destination")
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }

        public static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Console.WriteLine("=============== Training  model ===============");

            ITransformer model = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine("=============== End of training process ===============");
            return model;
        }
    }
}
