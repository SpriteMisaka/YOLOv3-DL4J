import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {

    // 每个边界框管理 (类别数 + (x,y,w,h,c)) 个元素。
    private static final int BOX_LENGTH = (Config.CLASSES.size() + 5);
    // x 位于 (x,y,w,h,c) 的第零个位置。
    private static final int X_POS = 0;
    // y 位于 (x,y,w,h,c) 的第一个位置。
    private static final int Y_POS = 1;
    // w 位于 (x,y,w,h,c) 的第二个位置。
    private static final int WIDTH_POS = 2;
    // h 位于 (x,y,w,h,c) 的第三个位置。
    private static final int HEIGHT_POS = 3;
    // c 位于 (x,y,w,h,c) 的第四个位置。
    private static final int CONFIDENCE_POS = 4;
    // 自第五个位置起为各类别的预测。
    private static final int CLASSES_POS = 5;

    static {
        // OpenCV 动态链接库。
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    /**
     * Sigmoid 函数。
     */
    private static double sigmoid(double x) {
        return  1 / (1 + Math.exp(-x));
    }
    /**
     * 输入图像并进行预处理。
     */
    private static INDArray loadInput(String imgPath) throws IOException {
        // 读入图像。
        File imgFile = new File(imgPath);
        INDArray input = new NativeImageLoader(Config.INPUT_HEIGHT, Config.INPUT_WIDTH, Config.CHANNELS)
                .asMatrix(imgFile);
        // 更改输入矩阵形状为[mini_batch, height, width, channel]以匹配网络。
        input = input.permute(0, 2, 3, 1);
        // 对输入矩阵进行归一化。
        new ImagePreProcessingScaler(0, 1).transform(input);
        return input;
    }
    /**
     * 载入模型。
     */
    private static ComputationGraph loadModel() throws IOException {
        String modelPath = new ClassPathResource(Config.MODEL_NAME).getFile().getPath();
        ComputationGraph model = null;
        try { model = KerasModelImport.importKerasModelAndWeights(modelPath); } catch (Exception ignored) { }
        return model;
    }
    /**
     * 对输出进行解码并归一化。
     */
    private static void decode(INDArray[] output) {
        // 对输出的所有尺度遍历。
        for (int scale = 0; scale < output.length; ++scale) {
            // 对所有网格遍历。
            for (int x = 0; x < output[scale].shape()[1]; ++x) {
                for (int y = 0; y < output[scale].shape()[2]; ++y) {
                    // 对边界框管理的各元素遍历。
                    for (int i = 0; i < output[scale].shape()[3]; ++i) {
                        // 当前访问元素的位置，所属的边界框，及其是否为边界框高度。
                        int[] position = new int[]{0, x, y, i};
                        int box = i / BOX_LENGTH, isHeight;
                        if (i % BOX_LENGTH == WIDTH_POS) { isHeight = 0; } else { isHeight = 1; }
                        // 当前访问元素的原始值和更新值。
                        double oldValue = output[scale].getDouble(position);
                        double newValue;
                        // 将边界框的宽高进行解码。
                        if (i == WIDTH_POS + isHeight + box * BOX_LENGTH) {
                            newValue = Math.exp(oldValue) *
                                    Config.ANCHORS[scale][box][isHeight] / (32 / Math.pow(2, scale));
                        }
                        // 将边界框的其他信息归一化。
                        else {
                            newValue = sigmoid(oldValue);
                        }
                        // 更新矩阵中的值。
                        output[scale].putScalar(position, newValue);
                    }
                }
            }
            // 更改输出矩阵形状为[mini_batch, box, x, y]以匹配网络。
            output[scale] = output[scale].permute(0, 3, 1, 2);
        }
    }
    /**
     * 绘制预测边界框。
     */
    private static void drawDetectedObjects(String imgPath, INDArray[] output) {
        // 读入图像。
        Mat img = Imgcodecs.imread(imgPath);
        // 获取预测的物体。
        List<BoundingBox> detectedObjects = getPredictedObjects(img, output);
        // 对所有预测的物体遍历。
        for (BoundingBox obj : detectedObjects) {
            // 绘制边界框。
            Imgproc.rectangle(img, obj.tl(), obj.br(), new Scalar(255, 0, 0), 2);
            Imgproc.putText(img, Config.CLASSES.get(obj.predictedClass) + String.format(" %.2f", obj.confidence),
                    obj.tl(), Imgproc.FONT_HERSHEY_PLAIN, 1.0, new Scalar(255, 0, 0), 2);
        }
        // 显示图像。
        HighGui.imshow("Image", img);
        HighGui.waitKey(0);
    }
    /**
     * 获取预测的物体。
     */
    public static List<BoundingBox> getPredictedObjects(Mat img, INDArray[] output) {
        // 候选框。
        List<BoundingBox> out = new ArrayList<>();
        // 对输出的所有尺度遍历。
        for (int scale = 0; scale < output.length; ++scale) {
            // 每一行的网格数。
            final int GRID_CELLS_PER_ROW = (int) Math.pow(2, scale) * 13;
            // 为便于操作，重新调整输出层形状。
            INDArray reshapedOutput = output[scale].dup('c')
                    .reshape(1, Config.ANCHORS.length, BOX_LENGTH, GRID_CELLS_PER_ROW, GRID_CELLS_PER_ROW);
            // 对所有网格遍历。
            for (int x = 0; (long) x < GRID_CELLS_PER_ROW; ++x) {
                for (int y = 0; (long) y < GRID_CELLS_PER_ROW; ++y) {
                    // 对每个网格的所有边界框遍历。
                    for (int box = 0; (long) box < Config.ANCHORS.length; ++box) {
                        // 获取置信度。
                        double confidence = reshapedOutput.getDouble(0, box, CONFIDENCE_POS, y, x);
                        // 若置信度高于阈值，则添加至候选框。
                        if (!(confidence < Config.CONFIDENCE_THRESHOLD)) {
                            // 获取边界框中心点坐标、宽度和高度。
                            double px = reshapedOutput.getDouble(0, box, X_POS, y, x) + x;
                            double py = reshapedOutput.getDouble(0, box, Y_POS, y, x) + y;
                            double pw = reshapedOutput.getDouble(0, box, WIDTH_POS, y, x);
                            double ph = reshapedOutput.getDouble(0, box, HEIGHT_POS, y, x);
                            // 计算边界框左上顶点和右下顶点的坐标。
                            double[] factor = {(double) img.width() / GRID_CELLS_PER_ROW, (double) img.height() / GRID_CELLS_PER_ROW};
                            int x1 = (int) ((px - pw / 2) * factor[X_POS]), y1 = (int) ((py - ph / 2) * factor[Y_POS]);
                            int x2 = (int) ((px + pw / 2) * factor[X_POS]), y2 = (int) ((py + ph / 2) * factor[Y_POS]);
                            // 计算物体所属类别。
                            INDArray softmax = reshapedOutput
                                    .get(NDArrayIndex.point(0), NDArrayIndex.point(box), NDArrayIndex.interval(CLASSES_POS, BOX_LENGTH),
                                            NDArrayIndex.point(y), NDArrayIndex.point(x)).dup();
                            int predictedClass = softmax.ravel().argMax().getInt(0);
                            // 添加边界框至候选框。
                            out.add(new BoundingBox(new Point(x1, y1), new Point(x2, y2), confidence, predictedClass));
                        }
                    }
                }
            }
        }
        // 非极大值抑制算法。
        out = BoundingBox.nms(out);
        return out;
    }

    public static void main(String[] args) throws IOException {
        // 输入图片。
        String imgPath = "src\\main\\resources\\example.jpg";
        INDArray input = loadInput(imgPath);
        // 读取模型。
        ComputationGraph model = loadModel();

        if (model != null) {
            // 得到预测结果。
            INDArray[] output = model.output(input);
            // 对输出进行解码并归一化。
            decode(output);
            // 绘制预测边界框。
            drawDetectedObjects(imgPath, output);
        }

        // 正常退出。
        System.exit(0);
    }
}
