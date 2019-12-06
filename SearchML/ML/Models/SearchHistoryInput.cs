using Microsoft.ML.Data;

namespace SearchML.ML.Models
{
    public class SearchHistoryInput
    {
        [ColumnName("source"), LoadColumn(0)]
        public string Source { get; set; }
        [ColumnName("destination"), LoadColumn(1)]
        public string Destination { get; set; }
    }

    public class SearchHistoryInputTransformed: SearchHistoryInput
    {
        public uint DestinationId { get; set; }
    }
}