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
        FloatNdArray inputArr = NdArrays.vectorOf(toArray(input));
        return TFloat32.tensorOf(inputArr);
    }

    public static TFloat32 toTensor2D (List<Double> input)
    {
        float[][] inputs2d = new float[1][];
        inputs2d[0] = toArray(input);
        FloatDataBuffer buf = DataBuffers.of(toArray(input));
        FloatNdArray inputArr = NdArrays.wrap(org.tensorflow.ndarray.Shape.of(1, input.size()), buf);
        return TFloat32.tensorOf(inputArr);
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
