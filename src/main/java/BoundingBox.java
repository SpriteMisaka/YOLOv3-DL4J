import org.opencv.core.Point;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class BoundingBox extends Rect implements Comparable<BoundingBox> {
    /**
     * 置信度。
     */
    public double confidence;
    /**
     * 所属类别。
     */
    public int predictedClass;
    /**
     * 构造函数。
     */
    BoundingBox(Point pt1, Point pt2, double confidence, int predictedClass) {
        super(pt1, pt2);
        this.confidence = confidence;
        this.predictedClass = predictedClass;
    }
    /**
     * 依据置信度比较边界框。
     */
    @Override
    public int compareTo(BoundingBox other) {
        return Double.compare(this.confidence, other.confidence);
    }
    /**
     * 计算并返回两个边界框的交并比。
     */
    public static double getIOU(BoundingBox box1, BoundingBox box2) {
        int width = Math.max(0,
                Math.min(box1.x + box1.width, box2.x + box2.width) - Math.max(box1.x, box2.x));
        int height = Math.max(0,
                Math.min(box1.y + box1.height, box2.y + box2.height) - Math.max(box1.y, box2.y));
        int intersection = width * height;
        int union = box1.width * box1.height + box2.width * box2.height - intersection;
        return (double)intersection / union;
    }
    /**
     * 非极大值抑制方法。
     */
    public static List<BoundingBox> nms(List<BoundingBox> boxes) {
        List<BoundingBox> result = new ArrayList<>();
        boxes.sort(Collections.reverseOrder());
        for (int i = 0; i < boxes.size(); ++i) {
            if (boxes.get(i).confidence < Config.CONFIDENCE_THRESHOLD) continue;
            result.add(boxes.get(i));
            for (int j = i + 1; j < boxes.size(); ++j) {
                double iou = getIOU(boxes.get(i), boxes.get(j));
                if (iou > Config.IOU_THRESHOLD) {
                    boxes.remove(j);
                    --j;
                }
            }
        }
        return result;
    }
}
