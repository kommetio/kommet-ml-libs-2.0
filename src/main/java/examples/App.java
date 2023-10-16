package examples;

import kommet.ml2.MLException;

import java.io.IOException;

public class App {

    public static void main(String[] args) throws MLException, IOException
    {
        //ModelLoadingAndPrediction.runPrediction();
        float pred = ModelLoadingAndPrediction.run2DPrediction(499, 11, "sample-input-1d.csv", "D:\\ml\\kmt-ml-libs-2.0\\model-tmp");
        System.out.println("Prediction: " + pred);
    }

}
