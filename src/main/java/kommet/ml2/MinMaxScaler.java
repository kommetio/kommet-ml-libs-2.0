package kommet.ml2;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.StringUtils;

public class MinMaxScaler extends Scaler
{
	private int rangeMin;
	private int rangeMax;
	private int featureCount;
	private List<Double> minValues;
	private List<Double> scaleValues;
	
	/**
	 * Loads the scaler from a file containing statistics.
	 * @param statsFile
	 * @param rangeMin
	 * @param rangeMax
	 * @return MinMaxScaler object
	 * @throws MLException
	 */
	public static MinMaxScaler load (String statsFile, int rangeMin, int rangeMax) throws MLException
	{
		MinMaxScaler scaler = new MinMaxScaler();
		scaler.setRangeMin(rangeMin);
		scaler.setRangeMax(rangeMax);
		
		BufferedReader br = null;
		
		try
		{
			br = new BufferedReader(new FileReader(statsFile));
			
			String minLine = null;

			// in newer version of the protocol, the first line will have the format
			// "scaler_type:min-max" or "scaler_type:robust"
			String firstLine = br.readLine();
			if (firstLine.startsWith("scaler_type:"))
			{
				List<String> scalerTypeParams = MiscUtils.splitAndTrim(firstLine, ":");
				String scalerType = scalerTypeParams.get(1);
				if (!"min-max".equals(scalerType))
				{
					throw new MLException("Incorrect scaler type '" + scalerType + "'. Expected 'min-max'");
				}

				// the min-max line will be the second one (after the one containing scaler type)
				minLine = br.readLine();
			}
			else
			{
				// the first line contains minimum values for scaling
				minLine = firstLine;
			}
			
			if (StringUtils.isEmpty(minLine))
			{
				throw new MLException("Scaler file does not contain min values");
			}
			
			List<String> sMinValues = MiscUtils.splitAndTrim(minLine, ",");
			List<Double> minValues = new ArrayList<Double>();
			for (String v : sMinValues)
			{
				minValues.add(Double.valueOf(v));
			}
			
			// the min values include the label, so we need to remove it
			minValues.remove(minValues.size() - 1);
			
			String scaleLine = br.readLine();
			
			if (StringUtils.isEmpty(scaleLine))
			{
				throw new MLException("Scaler file does not contain scale values");
			}
			
			List<String> sScaleValues = MiscUtils.splitAndTrim(scaleLine, ",");
			List<Double> scaleValues = new ArrayList<Double>();
			for (String v : sScaleValues)
			{
				scaleValues.add(Double.valueOf(v));
			}
			
			// the max values include the label, so we need to remove it
			scaleValues.remove(scaleValues.size() - 1);
			
			if (minValues.size() != scaleValues.size())
			{
				throw new MLException("Values min and scale have different sizes");
			}
			
			scaler.setFeatureCount(minValues.size());
			scaler.setMinValues(minValues);
			scaler.setScaleValues(scaleValues);
		}
		catch (Exception e)
		{
			e.printStackTrace();
			throw new MLException("Error initializing normalizer: " + e.getMessage());
		}
		finally
		{
			if (br != null)
			{
				try
				{
					br.close();
				}
				catch (IOException e)
				{
					throw new MLException("Error closing stream on file '" + statsFile + "'");
				}
			}
		}
		
		return scaler;
	}
	
	public List<Double> transform1d (List<Double> features)
	{
		List<Double> scaledFeatures = new ArrayList<Double>();
		
		int i = 0;
		for (Double f : features)
		{
			f *= this.scaleValues.get(i);
			f += this.minValues.get(i);
			scaledFeatures.add(f);
			i++;
		}
		
		return scaledFeatures;
	}

	public List<List<Double>> transform2d (List<List<Double>> observations)
	{
		List<List<Double>> scaledFeatures = new ArrayList<List<Double>>();

		for (int k = 0; k < observations.size(); k++)
		{
			List<Double> singleObservation = observations.get(k);
			List<Double> scaledObservation = new ArrayList<Double>();

			int featureIndex = 0;
			// iterate over features within this observation
			for (Double f : singleObservation)
			{
				f *= this.scaleValues.get(featureIndex);
				f += this.minValues.get(featureIndex);
				scaledObservation.add(f);
				featureIndex++;
			}

			scaledFeatures.add(scaledObservation);
		}

		return scaledFeatures;
	}

	public int getRangeMin()
	{
		return rangeMin;
	}

	public void setRangeMin(int rangeMin)
	{
		this.rangeMin = rangeMin;
	}

	public int getRangeMax()
	{
		return rangeMax;
	}

	public void setRangeMax(int rangeMax)
	{
		this.rangeMax = rangeMax;
	}

	public int getFeatureCount()
	{
		return featureCount;
	}

	public void setFeatureCount(int featureCount)
	{
		this.featureCount = featureCount;
	}

	public List<Double> getMinValues()
	{
		return minValues;
	}

	public void setMinValues(List<Double> minValues)
	{
		this.minValues = minValues;
	}

	public List<Double> getScaleValues()
	{
		return scaleValues;
	}

	public void setScaleValues(List<Double> scaleValues)
	{
		this.scaleValues = scaleValues;
	}
}
