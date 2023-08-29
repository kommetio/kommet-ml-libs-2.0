package examples;

import kommet.ml2.KerasModel;
import kommet.ml2.MLException;
import kommet.ml2.MiscUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class ModelLoadingAndPrediction {

    public static void runPrediction() {

        String kidderCols = "atr,boll100_placement,boll20_placement,boll100_spread,boll20_spread,close,close_diff,dc1000_placement,dc500_placement,dc100_placement,dc20_placement,dc5_placement,dc1000_spread,dc500_spread,dc100_spread,dc20_spread,dc5_spread,dc1000_sign,dc500_sign,dc100_sign,dc20_sign,dc5_sign,ema100,ema20,ema200,ema5,ema5_price,high,high_diff,low,low_diff,open,open_diff,rsi_14_0,rsi_14_1,rsi_14_10,rsi_14_2,rsi_14_3,rsi_14_4,rsi_14_5,rsi_14_6,rsi_14_7,rsi_14_8,rsi_14_9,rsi_50_0,rsi_50_1,rsi_50_2,rsi_50_3,rsi_50_4,rsi_50_5,ema_diff_20_100,ema_diff_100_200,ema_20_100_sign,ema_100_200_sign,gran_1,gran_2,gran_3,gran_4,gran_5,atr_span_1,atr_span_2,atr_span_3,atr_span_4,atr_span_5,atr_span_6,asset_type_0,asset_type_1,asset_type_2,gran,atr_10_50,atr_span";
        String tutuCols = "atr,boll100_placement,boll20_placement,boll100_spread,boll20_spread,close,close_diff,dc100_placement,dc20_placement,dc5_placement,dc100_spread,dc20_spread,dc5_spread,ema100,ema20,ema200,ema5,ema5_price,high,high_diff,low,low_diff,open,open_diff,rsi_14_0,rsi_14_1,rsi_14_10,rsi_14_2,rsi_14_3,rsi_14_4,rsi_14_5,rsi_14_6,rsi_14_7,rsi_14_8,rsi_14_9,rsi_50_0,rsi_50_1,rsi_50_2,rsi_50_3,rsi_50_4,rsi_50_5,atr_span,ema_diff_20_100,ema_diff_100_200,ema_20_100_sign,ema_100_200_sign,gran_1,gran_2,gran_3,gran_4,gran_5,gran";
        String bziuCols = "gran,rsi_5_0,rsi_5_1,rsi_5_5,rsi_5_10,rsi_10_0,rsi_10_1,rsi_10_5,rsi_10_10,rsi_14_0,rsi_14_1,rsi_14_5,rsi_14_10,rsi_50_0,rsi_50_1,rsi_50_5,rsi_50_10,boll20_placement,boll20_spread,boll100_placement,boll100_spread,dc20_placement,dc20_spread,dc100_placement,dc100_spread,macross_5_20_avglen,macross_5_20_currlen,macross_5_20_avglen_exceeded,ema_diff_5_20,ema_5_20_sign,macross_20_100_avglen,macross_20_100_currlen,macross_20_100_avglen_exceeded,ema_diff_20_100,ema_20_100_sign,atr_span";

        String kidderInputs = "0.0020051148595831327,-2.4730616158033633E-4,0.21620914039912595,14.258963007708694,8.771894837558458,1446.3410842224432,0.5934822109129477,0.19148936170212866,0.012800499531689621,0.01457518663348815,0.036123348017622424,0.0,18.75204296666447,15.974147239953787,14.029121506709233,5.660523608288395,1.0622832850797477,-1.0,-1.0,-1.0,-1.0,-1.0,1452.751689549398,1448.2796265365769,1454.9440826579475,1446.8078903984217,0.46680617597857815,1446.9894261335246,1.0722577760193066,1446.10169643989,0.4488520922872392,1446.9894261335246,1.0124108303810673,0.3431,0.36810000000000004,0.3699,0.389,0.3607,0.3455,0.29760000000000003,0.2999,0.34490000000000004,0.2858,0.287,0.401,0.40840000000000004,0.4143,0.4072,0.40340000000000004,0.39159999999999995,-4.472063012821112,-2.192393108549515,-1.0,-1.0,0,0,0,0,1,0,1,0,0,0,1,0.0,0.0,0.0,1,0.5,0.3";
        String tutuInputs = "0.005310290427199155,0.6368525584350766,0.16559539355171374,12.22467593197511,3.5083064468162313,4208.790518410153,0.5461094905738257,0.7847389558233342,0.0,0.0,11.722522685606062,3.0412649216471905,0.3107174687747168,4207.351237432063,4209.698378359828,4205.230072845225,4209.252828340966,0.4623099308134286,4209.3083808581105,0.8097485549891221,4208.404475494402,0.24480770267072544,4209.298965177239,0.3483801922625207,0.43829999999999997,0.47350000000000003,0.6665000000000001,0.4617,0.4775,0.4551,0.5478999999999999,0.501,0.5242,0.508,0.5409,0.5246,0.5346,0.5321,0.5367000000000001,0.5315,0.5574,15.0,2.3471409277656665,2.1211645868381743,1.0,1.0,0,0,0,0,1,1";
        String bziuInputs = "5,59.72,49.1,67.93,46.92,59.03,54.78,61.66,52.74,0.5847,0.5567,0.5995,0.5381,0.5608,0.5531,0.5626,54.31,0.7678039850381263,3.3706877166092872,0.8412604626122664,14.969074136308762,1.0413223140495935,2.734035442855265,1.0169133192389028,10.687593094797826,16.96,3,1.0,0.7309590626145869,1.0,81.71428571428571,72,0.0,2.630096905358278,1.0,10";

        List<String> features = MiscUtils.splitAndTrim(bziuCols, ",");

        try {

            // load model from disk
            KerasModel model = new KerasModel("D:\\ml\\kmt-ml-libs-2.0\\model-bziu", "D:\\ml\\kmt-ml-libs-2.0\\model-bziu\\stats.dat", features);

            // convert string inputs to double
            List<String> inputs = MiscUtils.splitAndTrim(bziuInputs, ",");
            List<Double> doubleInputs = new ArrayList<Double>();
            for (String i : inputs) {
                doubleInputs.add(Double.valueOf(i));
            }

            // run prediction
            float prediction = model.predict(doubleInputs);
            System.out.println("Prediction: " + prediction);
        } catch (MLException e) {
            e.printStackTrace();
        }
    }

    private static void getLibPath() {

        String javaLibPath = System.getProperty("java.library.path");
        Map<String, String> envVars = System.getenv();
        System.out.println(javaLibPath);

        /*for (String var : envVars.keySet()) {
            System.err.println("examining " + var);
            if (envVars.get(var).equals(javaLibPath)) {
                System.out.println(var);
            }
        }*/

    }

}
