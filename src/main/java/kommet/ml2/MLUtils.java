package kommet.ml2;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.types.TFloat32;

import java.util.ArrayList;
import java.util.List;

public class MLUtils
{
    /**
     * This method is to show how to convert the INDArray to a float array. This
     * is to provide some more examples on how to convert INDArray to types that
     * are more Java-centric.
     *
     * @param rowSlice
     * @return
     */
    public static float[] getFloatArrayFromSlice(INDArray rowSlice)
    {
        float[] result = new float[rowSlice.columns()];
        for (int i = 0; i < rowSlice.columns(); i++)
        {
            result[i] = rowSlice.getFloat(i);
        }
        return result;
    }

    public static TFloat32 toTensor (List<Double> input)
    {
        FloatNdArray inputArr = NdArrays.vectorOf(toArray1D(input));
        return TFloat32.tensorOf(inputArr);
    }

    /**
     * Converts the list of double values to a 2D tensor with shape [1, num-of-values]
     * @param input
     * @return
     */
    public static TFloat32 toTensor2D (List<Double> input)
    {
        FloatDataBuffer buf = DataBuffers.of(toArray1D(input));
        FloatNdArray inputArr = NdArrays.wrap(org.tensorflow.ndarray.Shape.of(1, input.size()), buf);
        return TFloat32.tensorOf(inputArr);
    }

    public static TFloat32 toTensor3D (List<List<Double>> input)
    {
        List<Double> listOneDim = new ArrayList<Double>();
        int dim1Size = input.size();
        int dim2Size = input.get(0).size();

        for (List<Double> observation : input)
        {
            listOneDim.addAll(observation);
        }

        FloatDataBuffer buf = DataBuffers.of(toArray1D(listOneDim));
        FloatNdArray inputArr = NdArrays.wrap(org.tensorflow.ndarray.Shape.of(1, dim1Size, dim2Size), buf);
        return TFloat32.tensorOf(inputArr);
    }

    private static float[] toArray1D(List<Double> list)
    {
        float[] array = new float[list.size()];
        for(int i = 0; i < list.size(); i++)
        {
            array[i] = list.get(i).floatValue();
        }
        return array;
    }

    private static float[][] toArray2D(List<List<Double>> list)
    {
        float[][] array = new float[list.size()][];

        int observationId = 0;
        for (List<Double> observation : list)
        {
            float[] obsArr = new float[observation.size()];
            for (int i = 0; i < observation.size(); i++)
            {
                obsArr[i] = observation.get(i).floatValue();
            }
            array[observationId++] = obsArr;
        }

        return array;
    }

    public static String listToString (List<Double> list)
    {
        List<String> newList = new ArrayList<String>();
        for (Double d : list)
        {
            newList.add(d != null ? d.toString() : "null");
        }
        return MiscUtils.implode(newList, ", ");
    }

    public static String getLibPath() {
        return System.getProperty("java.library.path");
    }
}
