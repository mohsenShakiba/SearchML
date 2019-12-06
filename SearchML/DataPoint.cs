using System.Numerics;

namespace SearchML
{
    public class DataPoint
    {
        public string Source { get; set; }
        public string Destination { get; set; }
    }

    public class TransformedData : DataPoint
    {
        public uint SourceCategory { get; set; }
        public float[] DestinationCategory { get; set; }
    }
}