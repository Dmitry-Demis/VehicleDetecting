using System.Drawing;
using Keras;
using Keras.Datasets;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Keras.Utils;
using Numpy;
using NumSharp;
using np = Numpy.np;
using Shape = Keras.Shape;
using System.Linq;
using OpenCvSharp;

namespace Helper
{
    internal class Program
    {
        static void Main(string[] args)
        {
            ////var trainDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Train\";
            ////var testDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Test\";
            ////var valDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Valid\";
            ////VehicleAvailability vehicle = new();
            ////vehicle.Train((440, 100, 3), trainDir, testDir, valDir, 20, 20, 400, 200, 200, classMode: "binary", outPath: @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Models\");
            //TrainSymbols();
            // input image dimensions
            int batch_size = 1;
            int num_classes = 22;
            int epochs = 24;
            int img_rows = 102, img_cols = 72;

            Shape input_shape = null;
            List<NDarray> x = new List<NDarray>();
            List<NDarray> y = new List<NDarray>();
            var trainDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Symbols\symbols\Train\";
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
                    var img = image.ToNDArray(flat:true);
                    img /= 255.0F;
                    Numpy.NDarray test = Numpy.np.array(img.ToArray<float>(), Numpy.np.float32);
                    test = test.reshape(72, 102, 1);
                    test = Numpy.np.expand_dims(test, 0);
                    x.Add(test);
                }
            }
            NDarray r = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
            NDarray l = np.array(new float[,] { { 1, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
           r =  Numpy.np.append(r, l);
            Console.WriteLine(r);
            x_train = x[0];
            y_train = y[0];
            for (int i = 1; i < x.Count; i++)
            {
                x_train =  Numpy.np.append(x_train, x[i]);
                y_train  = Numpy.np.append(y_train, y[i]);
               // Console.WriteLine(x_train);
            }

            x_train = x_train.reshape(38, 72, 102, 1);
            y_train = y_train.reshape(38, 1);
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
            int a = 5;
            // y_test = Util.ToCategorical(y_test, num_classes);

            // Build CNN model
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
                optimizer: new Adadelta(), metrics: new string[] { "accuracy" });
           // var x_test =
            model.Fit(x_train, y_train,
                        batch_size: batch_size,
                        epochs: epochs,
                        verbose: 1,
                        validation_data: new NDarray[] { x_test, y_test });
            var score = model.Evaluate(x_test, y_test, verbose: 1);
            Console.WriteLine($"Test loss: {score[0]}");
            Console.WriteLine($"Test accuracy: {score[1]*100.0:0.00}%");
        }
        public static void TrainSymbols()
        {
            var trainDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Symbols\symbols\Train\";
            var testDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Symbols\symbols\Test\";
            //var testDir = @"D:\HEI\BLOCK 4C\Diploma\DetectingVehicle\NeuralNet\Images\Test\test\";
            var valDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Symbols\symbols\Valid\";
            //CNN_ForCarDetecting((440, 100, 3), trainDir, testDir, valDir, 20, 20, 400, 200, 200, classMode: "binary");
            Numbers numbers = new Numbers();
            numbers.Train((72, 102, 1), trainDir, testDir, valDir, 1, 1, 44, 44, 44,
                @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Models\", 22);
        }
    }
}