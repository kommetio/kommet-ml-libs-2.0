package kommet.ml2;

import java.io.File;
import java.util.List;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;

import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;

public class KerasNNRunner
{
	public static void main(String[] args) throws MLException
	{
		//(new KerasNNRunner()).predict("C:\\Users\\krawiecr\\Desktop\\test\\model-oulu.h5", MiscUtils.toList(0.5, 0.5), "C:\\Users\\krawiecr\\Desktop\\test\\model-oulu-.gz");
    }
	
	public SavedModelBundle loadModel (String dir)
	{
		return SavedModelBundle.load(dir, "serve");
	}
	
	@SuppressWarnings("rawtypes")
	public float predict (SavedModelBundle model, List<Double> inputs, String scalerFile)
	{	
		long startDate = System.currentTimeMillis();
		Tensor x = NNUtils.toTensor(inputs);

		// running prediction
		TFloat32 outputTensor = (TFloat32)model.session().runner().feed("serving_default_input_1:0", x).fetch("StatefulPartitionedCall:0").run().get(0);
		float[][] output = StdArrays.array2dCopyOf(outputTensor);

		//float[][] output = model.session().runner().feed("serving_default_input_1:0", x).fetch("StatefulPartitionedCall:0").run().get(0).copyTo(new float[1][1]);
		System.out.println("duration: " + (System.currentTimeMillis() - startDate));
		return output[0][0];
	}
	
	public float predict (String dir, List<Double> inputs, String scalerFile) throws MLException
	{	
		return predict(loadModel(dir), inputs, scalerFile);
	}
	
	private static float[] toArray(List<Double> list)
	{
		float[] array = new float[list.size()];
		for(int i = 0; i < list.size(); i++)
		{
			array[i] = list.get(i).floatValue();
		}
		return array;
	}

	public double predict (String jsonFile, String weightsFile, List<Double> inputs, String scalerFile) throws MLException
	{
		try
		{
			/*SavedModelBundle load = SavedModelBundle.load(modelDir, "serve");
	        float[][] resultArray;
	        try (Graph g = load.graph()) {
	            try (Session s = load.session();
	                 Tensor result = s.runner().feed("data", data).fetch("prediction").run().get(0)) {
	                resultArray = result.copyTo(new float[10][1]);
	            }
	        }
	        load.close();
	        return resultArray;*/
			
			//MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(modelFile);
			//MultiLayerConfiguration config = KerasModelImport.importKerasSequentialConfiguration(modelFile);
			MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(jsonFile, weightsFile, false);
			
			NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0, 1);
			scaler.load(new File(scalerFile));
			
	
			INDArray features = Nd4j.zeros(inputs.size());
			
			int i = 0;
			for (Double input : inputs)
			{
				features.putScalar(new int[] {i}, input);
				i++;
			}
			
			scaler.transform(features);
			
			// get the prediction
			return model.output(features).getDouble(0);
		}
		catch (Exception e)
		{
			e.printStackTrace();
			throw new MLException("Error running model: " + e.getMessage());
		}
	}
}
