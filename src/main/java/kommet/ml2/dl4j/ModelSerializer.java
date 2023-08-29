package kommet.ml2.dl4j;

import java.io.File;
import java.io.IOException;

public class ModelSerializer
{
	public static MultiLayerNetwork restoreMultiLayerNetwork (File file) throws IOException
	{
		return new MultiLayerNetwork(org.deeplearning4j.util.ModelSerializer.restoreMultiLayerNetwork(file));
	}
}
