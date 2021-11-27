import java.io.*;
import java.util.*;

public class Config {
    /**
     * 配置文件。
     */
    private static final ResourceBundle BUNDLE = ResourceBundle.getBundle("config");
    /**
     * 目标识别的类别。
     */
    public static final List<String> CLASSES;
    /**
     * 模型文件名称。
     */
    public static final String MODEL_NAME = BUNDLE.getString("model_name");
    /**
     * 交并比阈值。
     */
    public static final double IOU_THRESHOLD =
            Double.parseDouble(BUNDLE.getString("iou_threshold"));
    /**
     * 置信度阈值。
     */
    public static final double CONFIDENCE_THRESHOLD =
            Double.parseDouble(BUNDLE.getString("confidence_threshold"));
    /**
     * 输入图像宽度。
     */
    public static final int INPUT_WIDTH =
            Integer.parseInt(BUNDLE.getString("input_width"));
    /**
     * 输入图像高度。
     */
    public static final int INPUT_HEIGHT =
            Integer.parseInt(BUNDLE.getString("input_height"));
    /**
     * 图像通道数。
     */
    public static final int CHANNELS =
            Integer.parseInt(BUNDLE.getString("channels"));
    /**
     * 预设边界框。
     */
    public static final int[][][] ANCHORS = new int[3][3][2];

    static {
        // 载入类别。
        {
            File file = new File("src\\main\\resources\\" + BUNDLE.getString("classes_name"));
            BufferedReader reader;
            ArrayList<String> lines = new ArrayList<>();
            try {
                reader = new BufferedReader(new FileReader(file));
                for (String currentLine; (currentLine = reader.readLine()) != null; ) {
                    lines.add(currentLine);
                }
            } catch (IOException ignored) { } finally {
                CLASSES = Collections.unmodifiableList(lines);
            }
        }
        // 载入预设边界框。
        {
            String[] scales = BUNDLE.getString("anchors").split(",");
            for (int i = 0; i < ANCHORS.length; ++i) {
                for (int j = 0; j < ANCHORS[i].length; ++j) {
                    for (int k = 0; k < ANCHORS[i][j].length; ++k) {
                        ANCHORS[ANCHORS.length - 1 - i][j][k] =
                                Integer.parseInt(scales[2 * i * ANCHORS.length + 2 * j + k]);
                    }
                }
            }
        }
    }
}
