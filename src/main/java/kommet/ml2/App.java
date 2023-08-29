package kommet.ml2;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class App {

    public static void main(String[] args) {

        //SavedModelBundle.load("D:\\ml\\kmt-ml-libs\\model_tutu\\pb", "serve");

        String kidderCols = "atr,boll100_placement,boll20_placement,boll100_spread,boll20_spread,close,close_diff,dc1000_placement,dc500_placement,dc100_placement,dc20_placement,dc5_placement,dc1000_spread,dc500_spread,dc100_spread,dc20_spread,dc5_spread,dc1000_sign,dc500_sign,dc100_sign,dc20_sign,dc5_sign,ema100,ema20,ema200,ema5,ema5_price,high,high_diff,low,low_diff,open,open_diff,rsi_14_0,rsi_14_1,rsi_14_10,rsi_14_2,rsi_14_3,rsi_14_4,rsi_14_5,rsi_14_6,rsi_14_7,rsi_14_8,rsi_14_9,rsi_50_0,rsi_50_1,rsi_50_2,rsi_50_3,rsi_50_4,rsi_50_5,ema_diff_20_100,ema_diff_100_200,ema_20_100_sign,ema_100_200_sign,gran_1,gran_2,gran_3,gran_4,gran_5,atr_span_1,atr_span_2,atr_span_3,atr_span_4,atr_span_5,atr_span_6,asset_type_0,asset_type_1,asset_type_2,gran,atr_10_50,atr_span";
        String tutuCols = "atr,boll100_placement,boll20_placement,boll100_spread,boll20_spread,close,close_diff,dc100_placement,dc20_placement,dc5_placement,dc100_spread,dc20_spread,dc5_spread,ema100,ema20,ema200,ema5,ema5_price,high,high_diff,low,low_diff,open,open_diff,rsi_14_0,rsi_14_1,rsi_14_10,rsi_14_2,rsi_14_3,rsi_14_4,rsi_14_5,rsi_14_6,rsi_14_7,rsi_14_8,rsi_14_9,rsi_50_0,rsi_50_1,rsi_50_2,rsi_50_3,rsi_50_4,rsi_50_5,atr_span,ema_diff_20_100,ema_diff_100_200,ema_20_100_sign,ema_100_200_sign,gran_1,gran_2,gran_3,gran_4,gran_5,gran";

        String kidderInputs = "0.0020051148595831327,-2.4730616158033633E-4,0.21620914039912595,14.258963007708694,8.771894837558458,1446.3410842224432,0.5934822109129477,0.19148936170212866,0.012800499531689621,0.01457518663348815,0.036123348017622424,0.0,18.75204296666447,15.974147239953787,14.029121506709233,5.660523608288395,1.0622832850797477,-1.0,-1.0,-1.0,-1.0,-1.0,1452.751689549398,1448.2796265365769,1454.9440826579475,1446.8078903984217,0.46680617597857815,1446.9894261335246,1.0722577760193066,1446.10169643989,0.4488520922872392,1446.9894261335246,1.0124108303810673,0.3431,0.36810000000000004,0.3699,0.389,0.3607,0.3455,0.29760000000000003,0.2999,0.34490000000000004,0.2858,0.287,0.401,0.40840000000000004,0.4143,0.4072,0.40340000000000004,0.39159999999999995,-4.472063012821112,-2.192393108549515,-1.0,-1.0,0,0,0,0,1,0,1,0,0,0,1,0.0,0.0,0.0,1,0.5,0.3";
        String tutuInputs = "0.06665165422607827,0.2106621553161209,0.23348403398883238,11.39029060673635,4.621777067267179,88.59495039643888,-0.07501689279969262,0.021001615508885147,0.019047619047621183,0.20000000000001183,9.287091328602141,3.150709497587157,0.22505067839907786,92.1055009254076,89.47488354560073,94.50368596456391,88.66716165744786,0.07221126100897833,88.59495039643888,0.25505743551895227,88.51993350363918,-0.2550574355189656,88.51993350363918,0.1650371641593291,0.38780000000000003,0.3791,0.36840000000000006,0.397,0.37270000000000003,0.3812,0.36619999999999997,0.3872,0.3803,0.4316,0.45039999999999997,0.42969999999999997,0.428,0.43229999999999996,0.42729999999999996,0.42950000000000005,0.4263,20.0,-2.630617379806871,-2.398185039156311,-1.0,-1.0,0,1,0,1,1,11";

        List<String> features = MiscUtils.splitAndTrim(tutuCols, ",");

        try {

            // load model from disk
            KerasModel model = new KerasModel("D:\\ml\\kmt-ml-libs-2.0\\model-tutu", "D:\\ml\\kmt-ml-libs-2.0\\model-tutu\\stats.dat", features);

            // convert string inputs to double
            List<String> inputs = MiscUtils.splitAndTrim(tutuInputs, ",");
            List<Double> doubleInputs = new ArrayList<Double>();
            for (String i : inputs) {
                doubleInputs.add(Double.valueOf(i));
            }

            // run prediction
            float prediction = model.predict(doubleInputs);
            System.out.println("Prediction: " + prediction);
        }
        catch (MLException e) {
            e.printStackTrace();
        }
    }

    private static void getLibPath() {

        String javaLibPath = System.getProperty("java.library.path");
        Map<String, String> envVars = System.getenv();
        //System.out.println(envVars.get("Path"));
        System.out.println(javaLibPath);

        /*for (String var : envVars.keySet()) {
            System.err.println("examining " + var);
            if (envVars.get(var).equals(javaLibPath)) {
                System.out.println(var);
            }
        }*/

    }

}
