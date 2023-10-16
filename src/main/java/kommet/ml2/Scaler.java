package kommet.ml2;

import java.util.List;

public abstract class Scaler
{
    public abstract List<Double> transform1d (List<Double> features);
    public abstract List<List<Double>> transform2d (List<List<Double>> observations);
}
