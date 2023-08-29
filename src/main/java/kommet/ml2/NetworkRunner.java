package kommet.ml2;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import kommet.ml2.dl4j.ModelSerializer;
import kommet.ml2.dl4j.MultiLayerNetwork;

/**
 * Runs a saved network.
 * @author krawiecr
 * @since 14/03/2018
 */
public class NetworkRunner
{
	private MultiLayerNetwork network;
	private RecordReaderDataSetIterator contextRecordReader;
	private static final int CONTEXT_BATCH_SIZE = 10;
	private DataNormalization dataNormalizer;
	
	/**
	 * Loads a saved network into the runner.
	 * @param file
	 * @throws NetworkRunnerException
	 */
	public void load (File file) throws NetworkRunnerException
	{
		try
		{
			this.network = ModelSerializer.restoreMultiLayerNetwork(file);
		}
		catch (IOException e)
		{
			throw new NetworkRunnerException("Error loading network: " + e.getMessage());
		}
	}
	
	public void load (MultiLayerNetwork model) throws NetworkRunnerException
	{
		this.network = model;
	}
	
	public void setInputContext (List<List<Double>> context, int labelIndex, int classCount)
	{
		Collection<Collection<Writable>> c = new ArrayList<>();
		
		for (List<Double> singleInput : context)
		{
			Collection<Writable> castInput = new ArrayList<Writable>();
			
			for (Double value : singleInput)
			{
				castInput.add(new DoubleWritable(value));
			}
			
			c.add(castInput);
		}

        CollectionRecordReader crr = new CollectionRecordReader(c);
        this.contextRecordReader = new RecordReaderDataSetIterator(crr, CONTEXT_BATCH_SIZE, labelIndex, classCount);
        
        // configure normalizer based on context data
        this.dataNormalizer = new NormalizerStandardize();
        this.dataNormalizer.fit(this.contextRecordReader);
	}
	
	public float[] predict (List<Double> input, int classCount) throws NetworkRunnerException
	{
		// iterate through the examples, but only take the last batch
		// because only the last example is the input we are checking
		
		DataSetIterator dataSetIterator = getDataSetIterator(input, input.size() - 1, classCount);
		dataSetIterator.setPreProcessor(this.dataNormalizer);
		
		INDArray features = dataSetIterator.next().getFeatures();//getFeatureMatrix();
        INDArray predicted = this.network.output(features);
		
		// get probabilities for the last prediction in the file, because this prediction contains result for our input
		return NNUtils.getFloatArrayFromSlice(predicted);
	}

	private static DataSetIterator getDataSetIterator(List<Double> input, int labelIndex, int classCount)
	{
		Collection<Collection<Writable>> c = new ArrayList<>();
		Collection<Writable> castInput = new ArrayList<Writable>();
			
		for (Double value : input)
		{
			castInput.add(new DoubleWritable(value));
		}
			
		c.add(castInput);

        CollectionRecordReader crr = new CollectionRecordReader(c);
        return new RecordReaderDataSetIterator(crr, CONTEXT_BATCH_SIZE, labelIndex, classCount);
	}
	
	public MultiLayerNetwork getModel()
	{
		return this.network;
	}
}
