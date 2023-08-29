package kommet.ml2.dl4j;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MultiLayerNetwork
{
	private org.deeplearning4j.nn.multilayer.MultiLayerNetwork model;

	public MultiLayerNetwork (MultiLayerConfiguration conf)
	{
		this.model = new org.deeplearning4j.nn.multilayer.MultiLayerNetwork(conf);
	}
	
	public MultiLayerNetwork (org.deeplearning4j.nn.multilayer.MultiLayerNetwork model)
	{
		this.model = model;
	}
	
	public INDArray output (INDArray input)
	{
		return this.model.output(input, false);
	}
}
