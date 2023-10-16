package examples;

import kommet.ml2.KerasModel;
import kommet.ml2.MLException;
import kommet.ml2.MiscUtils;
import kommet.ml2.ScalerType;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ModelLoadingAndPrediction {

    public static float run2DPrediction(int observationCount, int featureCount, String inputFilePath, String modelPath) throws MLException, IOException
    {
        String rawInput = Files.readString(Paths.get(inputFilePath), Charset.forName("UTF8"));

        // read all features * observations from a single line
        // the length of the line is num_features * num_observations
        List<String> rawInputItems = MiscUtils.splitAndTrim(rawInput, ",");

        // convert 1D input to 2D list of lists (num_observations * num_features)
        List<List<Double>> observations = new ArrayList<List<Double>>();
        for (int i = 0; i < observationCount; i++)
        {
            List<Double> observation = new ArrayList<>();
            for (int k = 0; k < featureCount; k++)
            {
                Double val = Double.parseDouble(rawInputItems.get(i * featureCount + k));
                observation.add(val);
            }
            observations.add(observation);
        }

        KerasModel model = new KerasModel(modelPath, "D:\\ml\\kmt-ml-libs-2.0\\model-tmp-scaler-stats-robust.dat", ScalerType.ROBUST, null);
        model.setVerbose(true);

        // return the prediction of the model
        return model.predict2d(observations, false);
    }

    public static void runPrediction() throws MLException
    {
        String kidderCols = "atr,boll100_placement,boll20_placement,boll100_spread,boll20_spread,close,close_diff,dc1000_placement,dc500_placement,dc100_placement,dc20_placement,dc5_placement,dc1000_spread,dc500_spread,dc100_spread,dc20_spread,dc5_spread,dc1000_sign,dc500_sign,dc100_sign,dc20_sign,dc5_sign,ema100,ema20,ema200,ema5,ema5_price,high,high_diff,low,low_diff,open,open_diff,rsi_14_0,rsi_14_1,rsi_14_10,rsi_14_2,rsi_14_3,rsi_14_4,rsi_14_5,rsi_14_6,rsi_14_7,rsi_14_8,rsi_14_9,rsi_50_0,rsi_50_1,rsi_50_2,rsi_50_3,rsi_50_4,rsi_50_5,ema_diff_20_100,ema_diff_100_200,ema_20_100_sign,ema_100_200_sign,gran_1,gran_2,gran_3,gran_4,gran_5,atr_span_1,atr_span_2,atr_span_3,atr_span_4,atr_span_5,atr_span_6,asset_type_0,asset_type_1,asset_type_2,gran,atr_10_50,atr_span";
        String tutuCols = "atr,boll100_placement,boll20_placement,boll100_spread,boll20_spread,close,close_diff,dc100_placement,dc20_placement,dc5_placement,dc100_spread,dc20_spread,dc5_spread,ema100,ema20,ema200,ema5,ema5_price,high,high_diff,low,low_diff,open,open_diff,rsi_14_0,rsi_14_1,rsi_14_10,rsi_14_2,rsi_14_3,rsi_14_4,rsi_14_5,rsi_14_6,rsi_14_7,rsi_14_8,rsi_14_9,rsi_50_0,rsi_50_1,rsi_50_2,rsi_50_3,rsi_50_4,rsi_50_5,atr_span,ema_diff_20_100,ema_diff_100_200,ema_20_100_sign,ema_100_200_sign,gran_1,gran_2,gran_3,gran_4,gran_5,gran";
        String bziuCols = "gran,rsi_5_0,rsi_5_1,rsi_5_5,rsi_5_10,rsi_10_0,rsi_10_1,rsi_10_5,rsi_10_10,rsi_14_0,rsi_14_1,rsi_14_5,rsi_14_10,rsi_50_0,rsi_50_1,rsi_50_5,rsi_50_10,boll20_placement,boll20_spread,boll100_placement,boll100_spread,dc20_placement,dc20_spread,dc100_placement,dc100_spread,macross_5_20_avglen,macross_5_20_currlen,macross_5_20_avglen_exceeded,ema_diff_5_20,ema_5_20_sign,macross_20_100_avglen,macross_20_100_currlen,macross_20_100_avglen_exceeded,ema_diff_20_100,ema_20_100_sign,atr_span";
        String imciCols = "atr,boll100_placement,boll20_placement,boll100_spread,boll20_spread,close,close_diff,dc100_placement,dc20_placement,dc5_placement,dc100_spread,dc20_spread,dc5_spread,ema100,ema20,ema200,ema5,ema5_price,high,high_diff,low,low_diff,open,open_diff,rsi_14_0,rsi_14_1,rsi_14_10,rsi_14_2,rsi_14_3,rsi_14_4,rsi_14_5,rsi_14_6,rsi_14_7,rsi_14_8,rsi_14_9,rsi_50_0,rsi_50_1,rsi_50_2,rsi_50_3,rsi_50_4,rsi_50_5,atr_span,ema_diff_20_100,ema_diff_100_200,ema_20_100_sign,ema_100_200_sign,gran_1,gran_2,gran_3,gran_4,gran_5,gran,atr_span_1,atr_span_2,atr_span_3,atr_span_4,atr_span_5,atr_span_6,atr_10_50,asset_type_0,asset_type_1,asset_type_2,hour_1,hour_2,hour_3,hour_4,hour_5,hour_6,weekday_1,weekday_2,weekday_3,weekday_4,macross_5_20_currlen,macross_5_20_avglen,macross_5_20_avglen_exceeded,macross_20_100_currlen,macross_20_100_avglen,macross_20_100_avglen_exceeded,boll_spread_sign,boll_placement_sign,dc_20_100_sign,close_diff_sign,rsi_14_diff,rsi_14_diff_2,rsi_14_50_diff,rsi_14_50_diff_sign";
        String saoirseCols = "atr,boll100_placement,boll20_placement,boll100_spread,boll20_spread,close,close_diff,dc100_placement,dc20_placement,dc5_placement,dc100_spread,dc20_spread,dc5_spread,ema100,ema20,ema200,ema5,ema5_price,high,high_diff,low,low_diff,open,open_diff,rsi_14_0,rsi_14_1,rsi_14_10,rsi_14_2,rsi_14_3,rsi_14_4,rsi_14_5,rsi_14_6,rsi_14_7,rsi_14_8,rsi_14_9,rsi_50_0,rsi_50_1,rsi_50_2,rsi_50_3,rsi_50_4,rsi_50_5,atr_span,ema_diff_20_100,ema_diff_100_200,ema_20_100_sign,ema_100_200_sign,gran_1,gran_2,gran_3,gran_4,gran_5,gran,atr_span_1,atr_span_2,atr_span_3,atr_span_4,atr_span_5,atr_span_6,atr_10_50,asset_type_0,asset_type_1,asset_type_2,hour_1,hour_2,hour_3,hour_4,hour_5,hour_6,weekday_1,weekday_2,weekday_3,weekday_4,boll_spread_sign,boll_placement_sign,dc_20_100_sign,close_diff_sign,rsi_14_diff,rsi_14_diff_2";

        String kidderInputs = "0.0020051148595831327,-2.4730616158033633E-4,0.21620914039912595,14.258963007708694,8.771894837558458,1446.3410842224432,0.5934822109129477,0.19148936170212866,0.012800499531689621,0.01457518663348815,0.036123348017622424,0.0,18.75204296666447,15.974147239953787,14.029121506709233,5.660523608288395,1.0622832850797477,-1.0,-1.0,-1.0,-1.0,-1.0,1452.751689549398,1448.2796265365769,1454.9440826579475,1446.8078903984217,0.46680617597857815,1446.9894261335246,1.0722577760193066,1446.10169643989,0.4488520922872392,1446.9894261335246,1.0124108303810673,0.3431,0.36810000000000004,0.3699,0.389,0.3607,0.3455,0.29760000000000003,0.2999,0.34490000000000004,0.2858,0.287,0.401,0.40840000000000004,0.4143,0.4072,0.40340000000000004,0.39159999999999995,-4.472063012821112,-2.192393108549515,-1.0,-1.0,0,0,0,0,1,0,1,0,0,0,1,0.0,0.0,0.0,1,0.5,0.3";
        String tutuInputs = "0.005310290427199155,0.6368525584350766,0.16559539355171374,12.22467593197511,3.5083064468162313,4208.790518410153,0.5461094905738257,0.7847389558233342,0.0,0.0,11.722522685606062,3.0412649216471905,0.3107174687747168,4207.351237432063,4209.698378359828,4205.230072845225,4209.252828340966,0.4623099308134286,4209.3083808581105,0.8097485549891221,4208.404475494402,0.24480770267072544,4209.298965177239,0.3483801922625207,0.43829999999999997,0.47350000000000003,0.6665000000000001,0.4617,0.4775,0.4551,0.5478999999999999,0.501,0.5242,0.508,0.5409,0.5246,0.5346,0.5321,0.5367000000000001,0.5315,0.5574,15.0,2.3471409277656665,2.1211645868381743,1.0,1.0,0,0,0,0,1,1";
        String bziuInputs = "5,59.72,49.1,67.93,46.92,59.03,54.78,61.66,52.74,0.5847,0.5567,0.5995,0.5381,0.5608,0.5531,0.5626,54.31,0.7678039850381263,3.3706877166092872,0.8412604626122664,14.969074136308762,1.0413223140495935,2.734035442855265,1.0169133192389028,10.687593094797826,16.96,3,1.0,0.7309590626145869,1.0,81.71428571428571,72,0.0,2.630096905358278,1.0,10";
        //String saoirseInputs = "7.242773989672856E-4,0.34844065317976675,0.5606375159930564,14.387223066759189,1.7532521886381665,1212.7673751140687,0.0,0.32013201320132734,1.153846153846318,1.0,12.550439946022149,0.717956960608476,0.8284118776252826,1214.4821877007528,1212.4511979141084,1215.8780617145514,1212.480192329825,-0.2871827842435437,1212.8778300310853,0.027613729254048345,1212.6845339263061,-0.4694333973209679,1212.7673751140687,-0.4694333973209679,0.5257999999999999,0.5257999999999999,0.5058,0.48,0.44689999999999996,0.4138,0.4474,0.4608,0.47859999999999997,0.47859999999999997,0.5038,0.4671,0.4671,0.4571,0.45030000000000003,0.44380000000000003,0.4508,10,-2.0309897866444993,-1.3958740137984746,-1.0,-1.0,0,0,1,1,1,7,0,0,1,0,1,0,0.0,0.0,0.0,0.0,0,0,0,0,1,1,0,1,0,1,1.0,0.0,0.0,0.0,0.07839999999999991,0.019999999999999907";
        String saoirseInputsRaw = "atr:7.242773989672856E-4,boll100_placement:0.34844065317976675,boll20_placement:0.5606375159930564,boll100_spread:14.387223066759189,boll20_spread:1.7532521886381665,close:1212.7673751140687,close_diff:0.0,dc100_placement:0.32013201320132734,dc20_placement:1.153846153846318,dc5_placement:1.0,dc100_spread:12.550439946022149,dc20_spread:0.717956960608476,dc5_spread:0.8284118776252826,ema100:1214.4821877007528,ema20:1212.4511979141084,ema200:1215.8780617145514,ema5:1212.480192329825,ema5_price:-0.2871827842435437,high:1212.8778300310853,high_diff:0.027613729254048345,low:1212.6845339263061,low_diff:-0.4694333973209679,open:1212.7673751140687,open_diff:-0.4694333973209679,rsi_14_0:0.5257999999999999,rsi_14_1:0.5257999999999999,rsi_14_10:0.5058,rsi_14_2:0.48,rsi_14_3:0.44689999999999996,rsi_14_4:0.4138,rsi_14_5:0.4474,rsi_14_6:0.4608,rsi_14_7:0.47859999999999997,rsi_14_8:0.47859999999999997,rsi_14_9:0.5038,rsi_50_0:0.4671,rsi_50_1:0.4671,rsi_50_2:0.4571,rsi_50_3:0.45030000000000003,rsi_50_4:0.44380000000000003,rsi_50_5:0.4508,atr_span:10,ema_diff_20_100:-2.0309897866444993,ema_diff_100_200:-1.3958740137984746,ema_20_100_sign:-1.0,ema_100_200_sign:-1.0,gran_1:0,gran_2:0,gran_3:1,gran_4:1,gran_5:1,gran:7,atr_span_1:0,atr_span_2:0,atr_span_3:1,atr_span_4:0,atr_span_5:1,atr_span_6:0,atr_10_50:0.0,asset_type_0:0.0,asset_type_1:0.0,asset_type_2:0.0,hour_1:0,hour_2:0,hour_3:0,hour_4:0,hour_5:1,hour_6:1,weekday_1:0,weekday_2:1,weekday_3:0,weekday_4:1,boll_spread_sign:1.0,boll_placement_sign:0.0,dc_20_100_sign:0.0,close_diff_sign:0.0,rsi_14_diff:0.07839999999999991,rsi_14_diff_2:0.019999999999999907";
        String imciInputsRaw = " atr:1.1362978403980859E-4,boll100_placement:0.033524523364270906,boll20_placement:0.31342074815356896,boll100_spread:15.079565032671109,boll20_spread:7.877001761363215,close:11968.516982515343,close_diff:0.9680560508811453,dc100_placement:0.15606936416184453,dc20_placement:0.3103448275862245,dc5_placement:0.0,dc100_spread:15.224881527486916,dc20_spread:7.656443311510072,dc5_spread:2.024117197295519,ema100:11975.187768756867,ema20:11970.602703279514,ema200:11978.813578692892,ema5:11969.529041113992,ema5_price:1.0120585986487367,high:11969.57304366176,high_diff:1.2320713374847387,low:11968.516982515343,low_diff:0.17601019107036503,open:11969.045013088551,open_diff:1.2320713374847387,rsi_14_0:0.40549999999999997,rsi_14_1:0.4354,rsi_14_10:0.4988,rsi_14_2:0.4465,rsi_14_3:0.4117,rsi_14_4:0.46,rsi_14_5:0.4628,rsi_14_6:0.5088,rsi_14_7:0.5907,rsi_14_8:0.5802,rsi_14_9:0.5238,rsi_50_0:0.40619999999999995,rsi_50_1:0.4174,rsi_50_2:0.4215,rsi_50_3:0.4071,rsi_50_4:0.42579999999999996,rsi_50_5:0.4269,atr_span:5,ema_diff_20_100:-4.585065477352916,ema_diff_100_200:-3.625809936026461,ema_20_100_sign:-1.0,ema_100_200_sign:-1.0,gran_1:0,gran_2:0,gran_3:0,gran_4:0,gran_5:0,gran:0,atr_span_1:0,atr_span_2:0,atr_span_3:0,atr_span_4:1,atr_span_5:0,atr_span_6:1,atr_10_50:1.0,asset_type_0:0.0,asset_type_1:0.0,asset_type_2:0.0,hour_1:0,hour_2:0,hour_3:0,hour_4:1,hour_5:0,hour_6:1,weekday_1:0,weekday_2:1,weekday_3:0,weekday_4:0,macross_5_20_currlen:59,macross_5_20_avglen:17.694779116465863,macross_5_20_avglen_exceeded:0.0,macross_20_100_currlen:101,macross_20_100_avglen:75.07272727272728,macross_20_100_avglen_exceeded:1.0,boll_spread_sign:1.0,boll_placement_sign:0.0,dc_20_100_sign:0.0,close_diff_sign:1.0,rsi_14_diff:-0.05730000000000002,rsi_14_diff_2:-0.09330000000000005,rsi_14_50_diff:-6.999999999999784E-4,rsi_14_50_diff_sign:0.0";
        //String imciInputs = "0.0487250549370636,0.2125702704337364,0.0889459483853923,7.720555037218186,6.42572617233275,396.51119993512543,-0.7585419872330661,0.1232287449392676,0.1420937295759502,0.2950191570881268,5.677571843835298,5.338321328441304,2.5711618008808754,399.5974765990356,398.4338042322576,400.6649561548243,396.9682543197488,0.4570543846234305,396.8229491988178,1.317392049796801,395.6321860467815,-1.1805014909536118,395.7713341710629,1.090198873425859,0.3359,0.2535,0.4217,0.3518,0.2928,0.38,0.3747,0.4048,0.4043,0.3902,0.3928999999999999,0.4311999999999999,0.4115999999999999,0.4484,0.437,0.4624,0.4614999999999999,3.0,-1.1636723667779676,-1.0674795557887429,-1.0,-1.0,0.0,1.0,0.0,1.0,0.0,10.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0,0.0,47.0,19.552380952380958,0.0,103.0,65.76666666666667,0.0,1.0,1.0,0.0,0.0,-0.0388,-0.08580000000000004,-0.09529999999999994"; //arrangeInputs(imciInputsRaw, imciCols);
        String imciInputs = arrangeInputs(imciInputsRaw, imciCols);
        //String saoirseInputs = arrangeInputs(saoirseInputsRaw, saoirseCols);

        System.out.println("Arranged inputs: " + imciInputs);

        List<String> features = MiscUtils.splitAndTrim(saoirseCols, ",");

        try
        {
            // load model from disk
            KerasModel model = new KerasModel("D:\\ml\\kmt-ml-libs-2.0\\model-evening", "D:\\ml\\kmt-ml-libs-2.0\\model-evening\\stats.dat", ScalerType.MINMAX, features);
            model.setVerbose(true);

            // convert string inputs to double
            List<String> inputs = MiscUtils.splitAndTrim(imciInputs, ",");
            List<Double> doubleInputs = new ArrayList<Double>();
            for (String i : inputs)
            {
                doubleInputs.add(Double.valueOf(i));
            }

            // run prediction
            float prediction = model.predict(doubleInputs, true);
            System.out.println("Prediction: " + prediction);
        }
        catch (MLException e)
        {
            e.printStackTrace();
        }
    }

    private static String arrangeInputs (String inputsRaw, String cols) throws MLException {
        Map<String, String> inputMap = new HashMap<String, String>();
        List<String> entries = MiscUtils.splitAndTrim(inputsRaw, ",");
        for (String e : entries)
        {
            List<String> featureParts = MiscUtils.splitAndTrim(e, ":");
            inputMap.put(featureParts.get(0), featureParts.get(1));
        }

        List<String> arrangedInputs = new ArrayList<String>();
        List<String> columns = MiscUtils.splitAndTrim(cols, ",");

        System.out.println("Arranging cols: " + columns.size());

        for (String col : columns)
        {
            if (!inputMap.containsKey(col))
            {
                throw new MLException("Raw input does not contain value for column " + col);
            }
            arrangedInputs.add(inputMap.get(col));
        }

        return MiscUtils.implode(arrangedInputs, ",");
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
