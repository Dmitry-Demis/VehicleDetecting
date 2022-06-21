using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AForge.Imaging.Filters;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Keras.PreProcessing.Image;
using Numpy;
using NumSharp;

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

        public void Train(string trainDir)
        {

        }
    }
    public class DataSetPreparing
    {

    }
}
