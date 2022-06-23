using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using Keras;
using Keras.Datasets;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Keras.Utils;
using Numpy;
using NumSharp;
using Shape = Keras.Shape;
using np = Numpy.np;
using System.Linq;
using AForge.Imaging;
using AForge.Imaging.Filters;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using Python.Runtime;
using Point = OpenCvSharp.Point;
using Size = OpenCvSharp.Size;


namespace Helper
{
    
    internal class Program
    {
        static class Processing
        {
            public static Bitmap FindLicensePlate(Bitmap image, string modelPath)
            {
                if (image == null) throw new ArgumentNullException(nameof(image));
                if (modelPath == null) throw new ArgumentNullException(nameof(modelPath));
                Rect[]? rects = null;
               // using ()
               
                   var classifier = new CascadeClassifier(modelPath);
                    var min = -14.0;
                    var max = Math.Abs(min);
                    var rotation = new RotateBicubic(min) { KeepSize = true };
                    Mat to = new();
                    while (min < max)
                    {
                        using (var result = rotation.Apply(image) ?? throw new ArgumentNullException())
                            to = result.ToMat();
                        rects = classifier.DetectMultiScale(to);
                        if (rects.Length == 0)
                            rotation.Angle = min += 0.2;
                        else break;
                    }
                    if (rects != null) to = to[new Rect(rects[0].X, rects[0].Y, rects[0].Width, rects[0].Height)];
                    to = to.Resize(new Size(85, 30));
                    image.Dispose();
                    classifier.Dispose();
                    GC.Collect();
                    return to.ToBitmap();
            }
            public static Bitmap HoughTransformation(Bitmap image)
            {
                if (image == null) throw new ArgumentNullException(nameof(image));
                var checker = new DocumentSkewChecker();
                if (image.PixelFormat != PixelFormat.Format8bppIndexed)
                    image = Grayscale.CommonAlgorithms.BT709.Apply(image);
                var angle = checker.GetSkewAngle(image);
                if (!(Math.Abs(Math.Abs(angle) - 90) < 0.1))
                {
                    var rotation = new RotateBicubic(-angle) { KeepSize = true };
                    image = rotation.Apply(image);
                }
                return image;
            }
            public static Bitmap LaplacianFilter(Bitmap image, int[,] filter)
            {
                if (image == null) throw new ArgumentNullException(nameof(image));
                if (filter == null) throw new ArgumentNullException(nameof(filter));
                if (image.PixelFormat != PixelFormat.Format8bppIndexed)
                    image = Grayscale.CommonAlgorithms.BT709.Apply(image);
            
                BrightnessCorrection b = new(40);
                image = b.Apply(image);
                ContrastCorrection c = new(30);
                image = c.Apply(image);

                var convolution = new Convolution(filter);
                image = convolution.Apply(image);
                return image;
            }
            public static Bitmap Crop(Bitmap image)
            {
                Mat img = image.ToMat();
                
                Mat imgCropped = img.Clone();
                img = img.Threshold(0, 255, ThresholdTypes.Otsu);
                Cv2.FindContours(img, out var contours, out var hierarchy, RetrievalModes.External,
                    ContourApproximationModes.ApproxSimple);
                Array.Sort(contours, (points, points1) =>
                {
                    var left = Cv2.ContourArea(points);
                    var right = Cv2.ContourArea(points1);
                    return !(left < right) ? -1 : 1;
                });
                var rect = Cv2.BoundingRect(contours[0]);
                var c = -1;
                var k = 0;
                var r = new Rect(rect.X + k, rect.Y + c, rect.Width - k, rect.Height - c );
                if (r.X + k + r.Width > imgCropped.Width)
                {
                    r.Width -= k;
                }

                if (r.X + k < 0)
                {
                    r.X = rect.X;
                }
                if (r.Y + c < 0)
                {
                    r.Y = rect.Y;
                }

               
                imgCropped = imgCropped[r];
                img.Dispose();
                image.Dispose();
                return imgCropped.ToBitmap();
            }
            public static void FindContours(Bitmap image2, Bitmap image, string outPath, int kernel)
            {
                
                int coeff = 5;
                Mat img = image.ToMat();
                Mat im = image2.ToMat();
               
                img = img.Resize(new Size(), coeff, coeff, InterpolationFlags.Cubic);
                im = im.Resize(new Size(), coeff, coeff, InterpolationFlags.Cubic);
                img = img.GaussianBlur(new Size(), 0.5, 0.5);
                if (kernel < 0) img = img.Threshold(0, 255, ThresholdTypes.Otsu);
                var rect_kern = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(2.5, 2.5));
                Mat dilation = new();
                //Cv2.Dilate(img, dilation, rect_kern);
                //Cv2.ImShow("i", dilation);
                //Cv2.WaitKey();
                Cv2.FindContours(dilation, out var contours, out var hierarchy, RetrievalModes.Tree,
                    ContourApproximationModes.ApproxSimple);
                //Cv2.DrawContours(im, contours, -1, Scalar.Cyan);
                //Cv2.ImShow("i", im);
                //Cv2.WaitKey();
                Array.Sort(contours, (points, points1) =>
                {
                    var left = Cv2.BoundingRect(points);
                    var right = Cv2.BoundingRect(points1);
                    return left.X < right.X ? -1 : 1;
                });
                List<Point[]> ct = new();
                for (int i = 0; i < contours.Length; i++)
                {
                    var rect = Cv2.BoundingRect(contours[i]);
                    var rect2 = (i != 0 && ct.Count != 0) ? Cv2.BoundingRect(ct[^1]) : Rect.Empty;
                    var area = Cv2.ContourArea(contours[i]);
                    if (rect.Width is > 50 or <= 12)
                        continue;
                    if (rect.Height <= 12)
                        continue;
                    if (i != 0 && rect.X + rect.Width < rect2.X + rect2.Width && rect.Y + rect.Height < rect2.Y + rect2.Height)
                        continue;
                    if (rect.Height * 1.0 / rect.Width is <= 0.8 or >= 2.6)
                        continue;
                    if (area < 300)
                        continue;
                    Cv2.Rectangle(im, Cv2.BoundingRect(contours[i]), Scalar.Red);
                    ct.Add(contours[i]);
                }
                //Cv2.ImShow("i", im);
                //Cv2.WaitKey();
                for (int j = 0; j < ct.Count; j++)
                {
                    var r = Cv2.BoundingRect(ct[j]);
                    Mat tmp = img[new Rect(r.X, r.Y, r.Width, r.Height)];
                    tmp = tmp.Resize(new Size(72, 102));
                    if (kernel < 0)
                        Cv2.BitwiseNot(tmp, tmp);
                    else
                        tmp = tmp.Threshold(0, 255, ThresholdTypes.Otsu);
                    var res = tmp.ToBitmap();
                    var t = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Result\";
                   res.Save(t+@$"{j}.png", ImageFormat.Png);
                   res.Dispose();
                   tmp.Dispose();

                }
                image.Dispose();
                image2.Dispose();
                img.Dispose();
                im.Dispose();
                // Cv2.DrawContours(img, contours, -1, Scalar.Red);
                //    Cv2.ImShow(nameof(img), img);
                //    Cv2.WaitKey();

            }
        }
        static void Main(string[] args)
        {
            //var trainDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Train\";
            //var testDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Test\";
            //var valDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Valid\";
            //VehicleAvailability vehicle = new();
            //vehicle.Train((440, 100, 3), trainDir, testDir, valDir, 20, 20, 400, 200, 200, classMode: "binary", outPath: @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Models\");
            //TrainSymbols();
            // input image dimensions
         Tr();
            //  REc();
            var check = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Check\";
            var files = Directory.GetFiles(check);
            const string modelPath = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Models\haarcascade_russian_plate_number.xml";
            var kernel1 = new[,] { { 0, -1, 0 }, { -1, 4, -1 }, { 0, -1, 0 } };
            var kernel2 = new[,] { { 0, 1, 0 }, { 1, -4, 1 }, { 0, 1, 0 } };
            var kernel3 = new[,] { { -1, -1, -1 }, { -1, 8, -1 }, { -1, -1, -1 } };
            var kernel4 = new[,] { { 1, 1, 1 }, { 1, -8, 1 }, { 1, 1, 1 } };
            var count = 0;
            var kernel = kernel4;
            Console.WriteLine("При использовании фильтра:");
            Console.WriteLine("[");
            for (int i = 0; i < 3; i++)
            {
               for (int j = 0; j < 3; j++)
                {
                    if (j==0)
                    {
                        Console.Write("[");
                    }

                    Console.Write($"{kernel[i, j]} ");
                    if (j == 2)
                    {
                        Console.WriteLine("]");
                    }
                    
                }
            }
            Console.WriteLine("]");
            if (!Directory.Exists(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Result"))
            {
                Directory.CreateDirectory(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Result");
            }
            for (int i = 0; i < files.Length; i++)
            {
                
                Bitmap image = new Bitmap(files[i]);

                image = Grayscale.CommonAlgorithms.BT709.Apply(image);
                // Sharpen s = new();
                //  image = s.Apply(image);
                image = Processing.FindLicensePlate(image, modelPath);
                image = Processing.HoughTransformation(image);
                image = Processing.Crop(image);
               // image = Processing.HoughTransformation(image);
                // image = Processing.HoughTransformation(image);
                image.Save(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\DEMO\check.png", ImageFormat.Png);

                Bitmap im = new Bitmap(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\DEMO\check.png");
                Bitmap im = new Bitmap(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\DEMO\check.png");
                image = Processing.LaplacianFilter(image, kernel);
                image.Save(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\DEMO\kernel.png", ImageFormat.Png);
                if (i != 0)
                {
                    Directory.Delete(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Result", true);
                    Directory.CreateDirectory(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Result");
                }
                Processing.FindContours(im, image, null, kernel[1, 1]);
                string s = Path.GetFileNameWithoutExtension(files[i]);
                var r = REc();
                if (s == r)
                {
                    count++;
                }
                image.Dispose();
                im.Dispose();

                Console.WriteLine($"expected = {s}, real  = {r}");

            }

            Console.WriteLine($"Точность распознавания равна {1.0*count/files.Length*100:0.00}%");
          

          
           // image.Save(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\DEMO\check.png", ImageFormat.Png);

            //float[] a = new float[22];
            //for (int j = 0; j < a.Length; j++)
            //{
            //    a[j] = j;
            //}

            //var n = new Numpy.NDarray(a);
            //n = Util.ToCategorical(n, a.Length);
        }

        public static void Tr()
        {
            int batch_size = 1;
            int num_classes = 22;
            int epochs = 80;
            int img_rows = 102, img_cols = 72;

            Shape input_shape = null;
            List<NDarray> x = new List<NDarray>();
            List<NDarray> xP = new List<NDarray>();
            List<NDarray> y = new List<NDarray>();
            var trainDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Symbols\symbols\Train\";
            var predict = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Result\";
            var array = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21";
            var split = array.Split(',');
            Numpy.NDarray x_train;
            Numpy.NDarray y_train;
            //var files = Directory.Get(trainDir);
            for (int i = 0; i < split.Length; i++)
            {
                var files = Directory.GetFiles(trainDir + split[i]);
                for (int j = 0; j < files.Length; j++)
                {
                    Bitmap image = new(files[j]);
                    y.Add(new NDarray(np.array(float.Parse(split[i]))));
                    var img = image.ToNDArray(flat: true);
                    img /= 255.0F;
                    Numpy.NDarray test = Numpy.np.array(img.ToArray<float>(), Numpy.np.float32);
                    test = test.reshape(72, 102, 1);
                    test = Numpy.np.expand_dims(test, 0);
                    x.Add(test);
                }
            }
            var filesP = Directory.GetFiles(predict);
            for (int i = 0; i < filesP.Length; i++)
            {
                Bitmap image = new(filesP[i]);
                var img = image.ToNDArray(flat: true);
                img /= 255.0F;
                Numpy.NDarray test = Numpy.np.array(img.ToArray<float>(), Numpy.np.float32);
                test = test.reshape(72, 102, 1);
                test = Numpy.np.expand_dims(test, 0);
                xP.Add(test);
            }
            x_train = x[0];
            y_train = y[0];
            for (int i = 1; i < x.Count; i++)
            {
                x_train = Numpy.np.append(x_train, x[i]);
                y_train = Numpy.np.append(y_train, y[i]);
                // Console.WriteLine(x_train);
            }

            x_train = x_train.reshape(x.Count, 72, 102, 1);
            y_train = y_train.reshape(x.Count, 1);
            //  = x_train.add(x.ToArray());

            // the data, split between train and test sets
            //var ((x_train, y_train), (x_test, y_test)) = MNIST.LoadData();

            //if (Backend.ImageDataFormat() == "channels_first")
            //{
            //    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols);
            //    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols);
            //    input_shape = (1, img_rows, img_cols);
            //}
            //else
            //{
            //    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1);
            //    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1);
            //    input_shape = (img_rows, img_cols, 1);
            //}

            //x_train = x_train.astype(np.float32);
            //x_test = x_test.astype(np.float32);
            //x_train /= 255;
            //x_test /= 255;
            //Console.WriteLine($"x_train shape: {x_train.shape}");
            //Console.WriteLine($"{x_train.shape[0]} train samples");
            //Console.WriteLine($"{x_test.shape[0]} test samples");

            // convert class vectors to binary class matrices
            //for (int i = 0; i < y.Count; i++)
            //{
            //    y[i] = Util.ToCategorical(y[i], num_classes);
            //}
            y_train = Util.ToCategorical(y_train, num_classes);
            var x_test = x_train;
            var y_test = y_train;
            var model = new Sequential();
            model.Add(new Conv2D(32, kernel_size: (3, 3).ToTuple(),
                                    activation: "relu",
                                    input_shape: input_shape));
            model.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu"));
            model.Add(new MaxPooling2D(pool_size: (2, 2).ToTuple()));
            model.Add(new Dropout(0.25));
            model.Add(new Flatten());
            model.Add(new Dense(128, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(num_classes, activation: "softmax"));
            model.Compile(loss: "categorical_crossentropy",
                optimizer: new Adam(), metrics: new string[] { "accuracy" });
            model.Fit(x_train, y_train,
                        batch_size: batch_size,
                        epochs: epochs,
                        verbose: 1,
                        validation_data: new NDarray[] { x_test, y_test });
            var score = model.Evaluate(x_test, y_test, verbose: 1);
            Console.WriteLine($"Test loss: {score[0]}");
            Console.WriteLine($"Test accuracy: {score[1] * 100.0:0.00}%");
            model.Save("resultModel.h5");
        }

        public static string REc()
        {
            var model2 = BaseModel.LoadModel("resultModel.h5");
            var predict = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Result\";
            List<NDarray> xP = new List<NDarray>();
            var filesP = Directory.GetFiles(predict);
            for (int i = 0; i < filesP.Length; i++)
            {
                Bitmap image = new(filesP[i]);
                var img = image.ToNDArray(flat: true);
                img /= 255.0F;
                Numpy.NDarray test = Numpy.np.array(img.ToArray<float>(), Numpy.np.float32);
                test = test.reshape(72, 102, 1);
                test = Numpy.np.expand_dims(test, 0);
                xP.Add(test);
            }
            var array = "0,1,2,3,4,5,6,7,8,9,A,B,C,E,H,K,M,O,P,T,X,У";
            var split = array.Split(',');
            Dictionary<string, string> dic = new Dictionary<string, string>(22);
            for (int i = 0; i < 22; i++)
            {
                double[] r = new double[22];
                r[i] = 1;
                string[] s = r.Select(z => z.ToString(CultureInfo.InvariantCulture)).ToArray();
                string a = String.Empty;
                for (int j = 0; j < s.Length; j++)
                {
                    a += s[j];
                }
                dic.Add(a, split[i]);
            }

            string d = default;
            for (int i = 0; i < xP.Count; i++)
            {
                var p = model2.Predict(xP[i], verbose:0);
                var s = (p.GetData<float>()).Select(r => Math.Round(r)).ToArray();
                string[] m = s.Select(z => z.ToString(CultureInfo.InvariantCulture)).ToArray();
                string a = String.Empty;
                for (int j = 0; j < s.Length; j++)
                {
                    a += m[j];
                }

                if (!dic.ContainsKey(a))
                {
                    d += "F";
                }
                else
                d += (dic[a]);
            }

            Console.WriteLine(d);
            return d;
        }
        public static void TrainSymbols()
        {
            var trainDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Symbols\symbols\Train\";
            var testDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Symbols\symbols\Test\";
            //var testDir = @"D:\HEI\BLOCK 4C\Diploma\DetectingVehicle\NeuralNet\Images\Test\test\";
            var valDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Symbols\symbols\Valid\";
            //CNN_ForCarDetecting((440, 100, 3), trainDir, testDir, valDir, 20, 20, 400, 200, 200, classMode: "binary");
            //Numbers numbers = new Numbers();
           // numbers.Train((72, 102, 1), trainDir, testDir, valDir, 1, 1, 44, 44, 44,
               // @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Models\", 22);
        }
    }
}