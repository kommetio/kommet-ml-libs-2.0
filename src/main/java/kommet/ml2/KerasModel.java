package kommet.ml2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;

public class KerasModel
{
	// the name of the input variable in the PB bundle
	private String inputName;
	
	// the name of the output variable in the PB bundle
	private String outputName;
	private SavedModelBundle bundle;
	private Scaler scaler;
	private boolean isVerbose;
	private String logPrefix;
	private List<String> featureNames;
	
	public KerasModel (String modelDir, String scalerPath, ScalerType scalerType, List<String> featureNames) throws MLException
	{
		this.inputName = "serving_default_input_1:0";
		this.featureNames = featureNames;
		this.outputName = "StatefulPartitionedCall:0";
		this.isVerbose = false;
		this.logPrefix = "[MR]";
		this.load(modelDir);
		
		if (!StringUtils.isEmpty(scalerPath))
		{
			switch (scalerType)
			{
				case MINMAX: this.scaler = MinMaxScaler.load(scalerPath, 0, 1); break;
			//this.loadMinMaxScaler(scalerPath, this.scalerMin, this.scalerMax);
				case ROBUST: this.scaler = RobustScaler.load(scalerPath); break;
				default: throw new MLException("Unsupported scaler type " + scalerType);
			}
		}
	}
	
	public float predict (List<Double> features)
	{
		return this.predict(features, true);
	}
	
	/**
	 * Runs the model and returns the prediction
	 * @param featureMap A string of format "<feature-name>:<feature-double-value>,..."
	 * @param isScale
	 * @return the value of the prediction returned by the model
	 * @throws MLException
	 */
	public float predict (String featureMap, boolean isScale) throws MLException
	{
		List<String> mapItems = MiscUtils.splitAndTrim(featureMap, ",");
		Map<String, Double> values = new HashMap<String, Double>();
		for (String item : mapItems)
		{
			List<String> itemParts = MiscUtils.splitAndTrim(item, "\\:");
			values.put(itemParts.get(0), itemParts.get(1) != null ? Double.valueOf(itemParts.get(1)) : null);
		}
		
		// build input features
		List<Double> features = new ArrayList<Double>();
		for (String feature : this.featureNames)
		{
			if (values.containsKey(feature))
			{
				features.add(values.get(feature));
			}
			else
			{
				throw new MLException("Feature map does not contain feature '" + feature + "'");
			}
		}
		
		return this.predict(features, isScale);
	}
	
	/**
	 * Runs a predictions
	 * @param features List of features in order specified when the KerasModel object was created
	 * @param isScale Whether min-max scaling should be applied to features
	 * @return The value of the prediction
	 */
	public float predict (List<Double> features, boolean isScale)
	{
		log("Predicting features: " + MLUtils.listToString(features));
		
		if (isScale)
		{
			log("Scaling features");
			features = this.scaler.transform1d(features);
			log("Scaled features: " + MLUtils.listToString(features));
		}

		// convert the feature values to a 2D tensor with shape [1, num-of-features]
		TFloat32 x = MLUtils.toTensor2D(features);

		log("Input shape: " + x.shape());
		log("Running prediction");
		TFloat32 outputTensor = (TFloat32)this.bundle.session().runner().feed(this.inputName, x).fetch(this.outputName).run().get(0);
		float[][] output = StdArrays.array2dCopyOf(outputTensor);
		log("Prediction: " + output[0][0]);
		return output[0][0];
	}
	
	/**
	 * Predicts based on a 2D input. The input contains N observations, and each observation consists of M features.
	 */
	public float predict2d (List<List<Double>> observations, boolean isScale)
	{
		if (isScale)
		{
			log("Scaling data");
			observations = this.scaler.transform2d(observations);

			/*for (List<Double> obs : observations)
			{
				for (Double val : obs)
				{
					System.out.print(val + ",");
				}
			}*/
		}

		// convert the feature values to a 2D tensor with shape [1, num-of-features]
		TFloat32 x = MLUtils.toTensor3D(observations);

		log("Input shape: " + x.shape());
		log("Running prediction");
		TFloat32 outputTensor = (TFloat32)this.bundle.session().runner().feed(this.inputName, x).fetch(this.outputName).run().get(0);
		float[][] output = StdArrays.array2dCopyOf(outputTensor);
		log("Prediction: " + output[0][0]);
		return output[0][0];
	}

	public void setVerbose (boolean verbose)
	{
		this.isVerbose = verbose;
	}

	public void setLogPrefix (String p)
	{
		this.logPrefix = p;
	}
	
	/*private static INDArray toINDArray (List<Double> inputs)
	{
		INDArray features = Nd4j.zeros(inputs.size());
		
		int i = 0;
		for (Double input : inputs)
		{
			features.putScalar(new int[] {i}, input);
			i++;
		}
		
		return features;
	}*/
	
	/*public void loadMinMaxScaler (String statsFile, int min, int max) throws RaimmeException
	{
		log("Loading scaler " + statsFile);
		this.scaler = new NormalizerMinMaxScaler(min, max);
		
		try
		{
			BufferedReader br = new BufferedReader(new FileReader(statsFile));
			
			String minLine = br.readLine();
			List<String> sMinValues = MiscUtils.splitAndTrim(minLine, ",");
			List<Double> minValues = new ArrayList<Double>();
			for (String v : sMinValues)
			{
				minValues.add(Double.valueOf(v));
			}
			
			// the min values include the label, so we need to remove it
			minValues.remove(minValues.size() - 1);
			
			String maxLine = br.readLine();
			List<String> sMaxValues = MiscUtils.splitAndTrim(maxLine, ",");
			List<Double> maxValues = new ArrayList<Double>();
			for (String v : sMaxValues)
			{
				maxValues.add(Double.valueOf(v));
			}
			
			// the max values include the label, so we need to remove it
			maxValues.remove(maxValues.size() - 1);
			
			for (int i = 0; i < minValues.size(); i++)
			{
				if (minValues.get(i).equals(maxValues.get(i)))
				{
					this.featuresWithZeroScalingRange.add(i);
				}
			}
		
			br.close();
			
			((NormalizerMinMaxScaler)scaler).setFeatureStats(toINDArray(minValues), toINDArray(maxValues));
		}
		catch (Exception e)
		{
			e.printStackTrace();
			throw new RaimmeException("Error initializing normalizer: " + e.getMessage());
		}
		
		log("Scaler initiated");
	}*/
	
	public void load (String dir)
	{
		log("Loading model " + dir);
		this.bundle = SavedModelBundle.load(dir, "serve");
		log("Model loaded");
	}
	
	public String getInputName()
	{
		return inputName;
	}
	
	public void setInputName(String inputName)
	{
		this.inputName = inputName;
	}
	
	public String getOutputName()
	{
		return outputName;
	}
	
	public void setOutputName(String outputName)
	{
		this.outputName = outputName;
	}
	
	private void log (String msg)
	{
		if (this.isVerbose)
		{
			System.out.println(this.logPrefix + " " + msg);
		}
	}
}
