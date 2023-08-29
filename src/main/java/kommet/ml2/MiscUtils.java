package kommet.ml2;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;



public class MiscUtils
{
    public static List<String> splitAndTrim(String items, String delimeter)
    {
        List<String> splitItems = new ArrayList<String>();
        String[] bits = items.split(delimeter);

        for (int i = 0; i < bits.length; i++)
        {
            splitItems.add(bits[i].trim());
        }

        return splitItems;
    }

    public static <T> String implode(Collection<T> items, String delimeter)
    {
        return implode(items, delimeter, null, null);
    }

    public static <T> String implode(Set<T> items, String delimeter)
    {
        return implode(items, delimeter, null, null);
    }

    public static <T> String implode(Set<T> items, String delimeter, String wrapper)
    {
        ArrayList<T> list = new ArrayList<T>();
        list.addAll(items);
        return implode (list, delimeter, wrapper, null);
    }

    public static <T> String implode(Set<T> items, String delimeter, String wrapper, String pre)
    {
        ArrayList<T> list = new ArrayList<T>();
        list.addAll(items);
        return implode (list, delimeter, wrapper, pre);
    }

    public static <T> String implode(Collection<T> items, String delimeter, String wrapper)
    {
        return implode(items, delimeter, wrapper, null);
    }

    public static <T> String implode(Collection<T> items, String delimeter, String wrapper, String pre)
    {
        ArrayList<T> list = new ArrayList<T>();
        list.addAll(items);
        return implode (list, delimeter, wrapper, pre);
    }

    public static <T> String implode(List<T> items, String delimeter, String wrapper)
    {
        return implode(items, delimeter, wrapper, null);
    }

    public static <T> String implode(List<T> items, String delimeter, String wrapper, String pre)
    {
        StringBuilder sb = new StringBuilder();

        if (items != null && !items.isEmpty())
        {
            for (T item : items)
            {
                sb.append(wrapper != null ? wrapper : "");
                sb.append(pre != null ? pre : "");
                sb.append(item);
                sb.append(wrapper != null ? wrapper : "");
                sb.append(delimeter != null ? delimeter : "");
            }
        }
        else
        {
            return items == null ? null : "";
        }

        return delimeter != null ? sb.toString().substring(0, sb.length() - delimeter.length()) : sb.toString();
    }
}

