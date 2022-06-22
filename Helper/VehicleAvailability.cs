using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AForge.Imaging.Filters;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Keras.PreProcessing.Image;
using Keras.Utils;
using Numpy;
using NumSharp;
using np = Numpy.np;
using Shape = Keras.Shape;

namespace Helper
{
    public class VehicleAvailability
    {
        private BaseModel? Model { get;  }

        public VehicleAvailability(string modelPath)
        {
            Model = BaseModel.LoadModel(modelPath ?? throw new ArgumentNullException(nameof(modelPath)));
        }
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Проверка совместимости платформы", Justification = "<Ожидание>")]
        public bool RecognizeVehicle(Bitmap image)
        {
            if (Model is null)
                throw new NullReferenceException("Use the method 'LoadModel'");
            if (image == null) throw new ArgumentNullException(nameof(image));
            if (Image.GetPixelFormatSize(image.PixelFormat) / 8 != 3)
                throw new ArgumentException("Only supported 24 pixel format");
            var img = image.ToNDArray(copy: false, flat: true);
            img /= 255.0F;
            Numpy.NDarray test = Numpy.np.array(img.ToArray<float>(), Numpy.np.float32);
            test = test.reshape(image.Width, image.Height, Image.GetPixelFormatSize(image.PixelFormat) / 8);
            test = Numpy.np.expand_dims(test, 0);
            var result = (double)Model.Predict(test, verbose: 0);
            result = Math.Round(result, 3);
            return (int)result == 1;
        }
        public void Train((int width, int height, int channels) image, string trainPath, string testPath, string valPath, int epoch, int batchSize,
           int trainSamples, int valSamples, int testSamples, string outPath, string loss = "binary_crossentropy", string classMode = "binary")
        {
            // var trainDir = @"D:\HEI\BLOCK 4C\Diploma\DetectingVehicle\NeuralNet\Images\Train\train\";
            //var testDir = @"D:\HEI\BLOCK 4C\Diploma\DetectingVehicle\NeuralNet\Images\Test\test\";
            //var valDir = @"D:\HEI\BLOCK 4C\Diploma\DetectingVehicle\NeuralNet\Images\Val\val\";
            //CNN_ForCarDetecting((440, 100, 3), trainDir, testDir, valDir, 20, 20, 400, 200, 200, classMode: "binary");
            var inputShape = new Keras.Shape(image.width, image.height, image.channels);
            var model = new Sequential();
            model.Add(new Conv2D(32, (3, 3).ToTuple(), input_shape: inputShape));
            model.Add(new Activation("relu"));
            model.Add(new MaxPooling2D((2, 2).ToTuple()));
            model.Add(new Conv2D(64, (3, 3).ToTuple()));
            model.Add(new Activation("relu"));
            model.Add(new MaxPooling2D((2, 2).ToTuple()));
            model.Add(new Flatten());
            model.Add(new Dense(5));
            model.Add(new Activation("relu"));
            model.Add(new Dropout(0.3));
            model.Add(new Dense(1));
            model.Add(new Activation("sigmoid"));
            model.Compile(new Adam(), loss: loss, metrics: new[] { "accuracy" });
            var dataGenerator = new ImageDataGenerator(rescale: (float?)(1 / 255.0));
            var trainGenerator = dataGenerator.FlowFromDirectory(trainPath, target_size: (image.width, image.height).ToTuple(), batch_size: batchSize, class_mode: classMode);
            var valGenerator = dataGenerator.FlowFromDirectory(valPath, target_size: (image.width, image.height).ToTuple(), batch_size: batchSize, class_mode: classMode);
            var testGenerator = dataGenerator.FlowFromDirectory(testPath, target_size: (image.width, image.height).ToTuple(), batch_size: batchSize, class_mode: classMode);
            model.FitGenerator(trainGenerator, steps_per_epoch: trainSamples / batchSize, epochs: epoch,
                validation_data: valGenerator, validation_steps: valSamples / batchSize);
            var scores = model.EvaluateGenerator(testGenerator, testSamples / batchSize);
            var acc = scores[1] * 100;
            Console.WriteLine($"Точность на тестовых данных равна: {acc:0.00}%");
            model.Save(outPath + $@"\carEmpty_{acc:0.00}.h5");
        }
    }
    class NumberDetector
    {
        private BaseModel? Model { get; }
        public NumberDetector(string? modelPath) => Model = BaseModel.LoadModel(modelPath);

        public void Train(string trainDir, Size size, int numClasses = 22,  int epochs = 80, int batchSize = 1)
        {
            int batch_size = 1;
            int num_classes = 22;
           
            int img_rows = 102, img_cols = 72;

            Shape input_shape = null;
            List<NDarray> x = new List<NDarray>();
            List<NDarray> xP = new List<NDarray>();
            List<NDarray> y = new List<NDarray>();
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
            Console.WriteLine($"Test accuracy: {score[1] * 100.0:0.00}%");
            model.Save("resultModel.h5");
        }

        public string Recongize(string predict, string classes, int classesCount = 22, char separator = ',')
        {
            var xP = new List<NDarray>();
            var files = Directory.GetFiles(predict);
            foreach (var file in files)
            {
                Bitmap image = new(file);
                var img = image.ToNDArray(flat: true);
                img /= 255.0F;
                NDarray converted = np.array(img.ToArray<float>(), np.float32);
                converted = converted.reshape(72, 102, 1);
                converted = np.expand_dims(converted, 0);
                xP.Add(converted);
            }
            var split = classes.Split(',');
            var dic = new Dictionary<string, string>(classesCount);
            for (var i = 0; i < classesCount; i++)
            {
                var r = new double[classesCount];
                r[i] = 1;
                var s = r.Select(z => z.ToString(CultureInfo.InvariantCulture)).ToArray();
                var a = String.Empty;
                for (var j = 0; j < s.Length; j++)
                {
                    a += s[j];
                }
                dic.Add(a, split[i]);
            }

            string d = default;
            for (var i = 0; i < xP.Count; i++)
            {
                var p = Model.Predict(xP[i]);
                var s = (p.GetData<float>()).Select(r => Math.Round(r)).ToArray();
                string[] m = s.Select(z => z.ToString(CultureInfo.InvariantCulture)).ToArray();
                var a = String.Empty;
                for (var j = 0; j < s.Length; j++)
                {
                    a += m[j];
                }
                d += (dic[a]);
            }

            Console.WriteLine(d);
            return null;
        }
    }
    public class DataSetPreparing
    {

    }
}
