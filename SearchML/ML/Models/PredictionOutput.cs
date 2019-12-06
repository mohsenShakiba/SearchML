using Microsoft.ML.Data;

namespace SearchML.ML.Models
{
    public class PredictionOutput
    {
        [ColumnName("PredictedLabel")]
        public string Prediction { get; set; }
        public float[] Score { get; set; }
    }
}