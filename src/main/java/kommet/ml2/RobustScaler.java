package kommet.ml2;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.StringUtils;

/**
 * Java implementation of sklearn.RobustScaler
 */
public class RobustScaler extends Scaler
{
    // center (median) values
    private List<Double> centerValues;
    private int featureCount;
    private List<Double> scaleValues;

    /**
     * Loads the scaler from a file containing statistics.
     * @param statsFile
     * @return MinMaxScaler object
     * @throws MLException
     */
    public static RobustScaler load (String statsFile) throws MLException
    {
        RobustScaler scaler = new RobustScaler();
        BufferedReader br = null;

        try
        {
            br = new BufferedReader(new FileReader(statsFile));

            // in newer version of the protocol, the first line will have the format
            // "scaler_type:min-max" or "scaler_type:robust"
            String scalerTypeList = br.readLine();

            List<String> scalerTypeParams = MiscUtils.splitAndTrim(scalerTypeList, ":");
            String scalerType = scalerTypeParams.get(1);
            if (!"robust".equals(scalerType))
            {
                throw new MLException("Incorrect scaler type '" + scalerType + "'. Expected 'robust'");
            }

            // feature names - not used
            String featureLine = br.readLine();
            // remove the last feature which is in fact the label
            int featureCount = MiscUtils.splitAndTrim(featureLine, ",").size() - 1;

            String centerLine = br.readLine();

            if (StringUtils.isEmpty(centerLine))
            {
                throw new MLException("Scaler file does not contain center values");
            }

            List<String> sCenterValues = MiscUtils.splitAndTrim(centerLine, ",");
            List<Double> centerValues = new ArrayList<>();
            for (String v : sCenterValues)
            {
                centerValues.add(Double.valueOf(v));
            }

            // the center (median) values include the label, so we need to remove it
            centerValues.remove(centerValues.size() - 1);

            String scaleLine = br.readLine();

            if (StringUtils.isEmpty(scaleLine))
            {
                throw new MLException("Scaler file does not contain scale values");
            }

            List<String> sScaleValues = MiscUtils.splitAndTrim(scaleLine, ",");
            List<Double> scaleValues = new ArrayList<>();
            for (String v : sScaleValues)
            {
                scaleValues.add(Double.valueOf(v));
            }

            // the max values include the label, so we need to remove it
            scaleValues.remove(scaleValues.size() - 1);

            if (centerValues.size() != scaleValues.size() || centerValues.size() != featureCount)
            {
                throw new MLException("Values center and/or scale have different sizes than the number of features: " + centerValues.size() + ", " + scaleValues.size() + ", " + featureCount);
            }

            scaler.setFeatureCount(featureCount);
            scaler.setCenterValues(centerValues);
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
            f *= (f - this.centerValues.get(i)) / this.scaleValues.get(i);
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
                f *= (f - this.centerValues.get(featureIndex)) / this.scaleValues.get(featureIndex);
                scaledObservation.add(f);
                featureIndex++;
            }

            scaledFeatures.add(scaledObservation);
        }

        return scaledFeatures;
    }

    public int getFeatureCount()
    {
        return featureCount;
    }

    public void setFeatureCount(int featureCount)
    {
        this.featureCount = featureCount;
    }

    public List<Double> getCenterValues()
    {
        return centerValues;
    }

    public List<Double> getScaleValues()
    {
        return scaleValues;
    }

    public void setScaleValues(List<Double> scaleValues)
    {
        this.scaleValues = scaleValues;
    }

    public void setCenterValues(List<Double> centerValues)
    {
        this.centerValues = centerValues;
    }
}
