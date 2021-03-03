package mlAPI.learners.classification.nn

import ControlAPI.LearnerPOJO
import org.nd4j.linalg.dataset.DataSet
import mlAPI.learners.{Learner, SGDUpdate}
import mlAPI.learners.classification.Classifier
import mlAPI.math.{LabeledPoint, LearningPoint}
import mlAPI.parameters.utils.{ParameterDescriptor, SerializableParameters}
import mlAPI.parameters.{DLParams, LearningParameters}
import mlAPI.utils.Parsing
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.{Model, OptimizationAlgorithm}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ActivationLayer, BatchNormalization, ConvolutionLayer, DenseLayer}
import org.deeplearning4j.nn.conf.layers.{DropoutLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{ConvolutionMode, MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.annotation.tailrec
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._
import scala.collection.mutable

/**
 * A Neural Network learner that utilizes the Deeplearning4j framework.
 *
 * @param conf The configuration of the Neural Network.
 */
case class NeuralNetwork(var conf: MultiLayerConfiguration,
                         var inputShape: Array[Int],
                         var numOfClasses: Int)
  extends Classifier with SGDUpdate with Serializable {

  override protected var miniBatchSize: Int = 64
  override protected val parallelizable: Boolean = true
  override protected var targetLabel: Double = 1.0
  override var learningRate: Double = 0.001D
  private var parametersSize: Array[Int] = Array[Int]()
  private val batchX: ListBuffer[Double] = ListBuffer[Double]()
  private val batchY: ListBuffer[Double] = ListBuffer[Double]()
  private val weights: DLParams = new DLParams()
  private var u: INDArray = _
  var NN: MultiLayerNetwork = new MultiLayerNetwork(conf)
  NN.init()

  def configure(conf: MultiLayerConfiguration, inputShape: Array[Int], numOfClasses: Int): Unit = {
    this.conf = conf
    this.inputShape = inputShape
    this.numOfClasses = numOfClasses
    NN = new MultiLayerNetwork(conf)
    NN.init()
  }

  //////////////////////////////////////// Setters and Getters for the parameters. /////////////////////////////////////

  def setParams(params: INDArray): Unit = NN.setParams(params)

  def setSerializedParams(sParams: (Array[Double], Array[Int], Char)): Unit =
    NN.setParams(Nd4j.create(sParams._1, sParams._2, sParams._3))

  def getParams: INDArray = NN.params()

  override def getParameters: Option[LearningParameters] = Some(DLParams(NN.params()))

  override def setParameters(params: LearningParameters): Learner = {
    NN.setParams(params.asInstanceOf[DLParams].parameters)
    this
  }

  def getNN: Model = NN

  def getConfig: MultiLayerConfiguration = conf

  def setNN(nn: MultiLayerNetwork): Unit = this.NN = nn

  ///////////////////////////////////////////////////// Auxiliary //////////////////////////////////////////////////////

  def jsonConf: String = getConfig.toJson

  def confFromJson(jsonConf: String, inputShape: Array[Int], numOfClasses: Int): Unit = {
    try {
      configure(MultiLayerConfiguration.fromJson(jsonConf), inputShape, numOfClasses)
    } catch {
      case _: Throwable => throw new RuntimeException("Error while loading a json Multilayer configuration.")
    }
  }

  def yamlConf: String = getConfig.toYaml

  def confFromYaml(yamlConf: String, inputShape: Array[Int], numOfClasses: Int): Unit = {
    try {
      configure(MultiLayerConfiguration.fromYaml(yamlConf), inputShape, numOfClasses)
    } catch {
      case _: Throwable => throw new RuntimeException("Error while loading a yaml Multilayer configuration.")
    }
  }

  def getParametersInfo: NNParametersInfo = NNParametersInfo(this.NN)

  @tailrec
  private def concatFeatures(batch: ListBuffer[LearningPoint], result: ListBuffer[Double]): Array[Double] = {
    if (batch.isEmpty)
      result.toArray
    else
      concatFeatures(batch.tail, result ++ batch.head.getNumericVector.toArray)
  }

  @tailrec
  private def concatPoints(batch: ListBuffer[LearningPoint], features: ListBuffer[Double], labels: ListBuffer[Double])
  : (Array[Double], Array[Double]) = {
    if (batch.isEmpty)
      (features.toArray, labels.toArray)
    else {
      val point = batch.head.asInstanceOf[LabeledPoint]
      concatPoints(batch.tail,
        features ++ batch.head.getNumericVector.toArray,
        labels ++ (for (i <- 0 to 9) yield { if (i * 1.0 == point.getLabel - 1.0) 1.0 else 0.0 }).toArray
      )
    }
  }

  override def predict(data: LearningPoint): Option[Double] = {
    val dataPoint = Nd4j.create(data.numericVector.toArray, inputShape, 'c')
    try {
      Some(NN.output(dataPoint, false).toDoubleVector.zipWithIndex.maxBy(_._1)._2.toDouble)
    } catch {
      case _: Throwable => None
    }
  }

  override def predict(batch: ListBuffer[LearningPoint]): Array[Option[Double]] = {
    try {
      NN.output(
        Nd4j.create(concatFeatures(batch, ListBuffer[Double]()), inputShape.updated(0, batch.length), 'c'),
        false
      ).argMax(1).toDoubleVector.map(x => Some(x))
    } catch {
      case _: Throwable => Array[Option[Double]](None)
    }
  }

  def constructMiniBatch(data: LearningPoint): Unit = {
    val dataPoint = data.asInstanceOf[LabeledPoint]
    batchX ++= dataPoint.getNumericVector.toArray
    batchY ++= (for (i <- 0 to 9) yield { if (i * 1.0 == dataPoint.getLabel - 1.0) 1.0 else 0.0 }).toArray
  }

  def getMiniBatch: DataSet = {
    val X = Nd4j.create(batchX.toArray, inputShape.updated(0, miniBatchSize))
    val Y = Nd4j.create(batchY.toArray, Array(miniBatchSize, numOfClasses))
    batchX.clear()
    batchY.clear()
    new DataSet(X, Y)
  }

  def fitMiniBatch(): Unit = {
    if (batchY.length == miniBatchSize * numOfClasses) {
      val it = getMiniBatch.dataSetBatches(miniBatchSize).iterator()
      while (it.hasNext)
        NN.fit(it.next())
    }
  }

  def fitMiniBatchLoss(): Double = {
    var loss: Double = 0.0
    if (batchY.length == miniBatchSize * numOfClasses) {
      val it = getMiniBatch.dataSetBatches(miniBatchSize).iterator()
      while (it.hasNext) {
        NN.fit(it.next())
        loss += NN.score()
      }
    }
    loss
  }

  override def fit(data: LearningPoint): Unit = {
    constructMiniBatch(data)
    fitMiniBatch()
  }

  override def fitLoss(data: LearningPoint): Double = {
    constructMiniBatch(data)
    fitMiniBatchLoss()
  }

  override def fit(batch: ListBuffer[LearningPoint]): Unit = {
    fitLoss(batch)
    ()
  }

  override def fitLoss(batch: ListBuffer[LearningPoint]): Double = (for (point <- batch) yield fitLoss(point)).sum

  override def score(testSet: ListBuffer[LearningPoint]): Double = {
    if (testSet.nonEmpty) {
      val (testX, testY) = concatPoints(testSet, ListBuffer[Double](), ListBuffer[Double]())
      val tX = Nd4j.create(testX, inputShape.updated(0, testSet.length), 'c')
      val tY = Nd4j.create(testY, Array(testSet.length, numOfClasses), 'c')
//      println(tX.shape().mkString("Array(", ", ", ")"))
//      println(tY.shape().mkString("Array(", ", ", ")"))
      val test = new ListDataSetIterator(new DataSet(tX, tY).asList())
      val eval: Evaluation = NN.evaluate(test)
//      println(tY.getRows(0,1,3,4))
//      println(NN.output(tX).getRows(0,1,3,4))
      eval.f1()
    } else 0.0D
  }

  override def setLearningRate(lr: Double): Unit = {
    super.setLearningRate(lr)
    NN.setLearningRate(lr)
  }

  override def setHyperParametersFromMap(hyperParameterMap: mutable.Map[String, AnyRef]): Learner = {
    for ((hyperparameter, value) <- hyperParameterMap) {
      hyperparameter match {
        case "inputShape" =>
          try {
            inputShape = value.asInstanceOf[java.util.List[Double]].asScala.toArray.map(_.toInt)
          } catch {
            case e: Exception =>
              println("Error while trying to update the inputShape hyper parameter of the Neural Network classifier.")
              e.printStackTrace()
          }
        case "numOfClasses" =>
          try {
            numOfClasses = Parsing.IntegerParsing(hyperParameterMap, "numOfClasses", 10)
          } catch {
            case e: Exception =>
              println("Error while trying to update the inputShape hyper parameter of the Neural Network classifier.")
              e.printStackTrace()
          }
        case "parametersSize" =>
          try {
            parametersSize = value.asInstanceOf[java.util.List[Double]].asScala.toArray.map(_.toInt)
          } catch {
            case e: Exception =>
              println("Error while trying to update the parametersSize hyper parameter of the Neural Network classifier.")
              e.printStackTrace()
          }
        case "miniBatchSize" =>
          miniBatchSize = Parsing.IntegerParsing(hyperParameterMap, "miniBatchSize", 64)
          println(miniBatchSize + " " + getMiniBatchSize)
        case "learningRate" =>
          try {
            setLearningRate(Parsing.DoubleParsing(hyperParameterMap, "learningRate", 0.001D))
          } catch {
            case e: Exception =>
              println("Error while trying to update the learningRate hyper parameter of the Neural Network classifier.")
              e.printStackTrace()
          }
        case "batchX" =>
          try {
            batchX.clear()
            for (v: Double <- value.asInstanceOf[java.util.List[Double]].asScala) batchX append v
          } catch {
            case e: Exception =>
              println("Error while trying to update the batchX hyper parameter of the Neural Network classifier.")
              e.printStackTrace()
          }
        case "batchY" =>
          try {
            batchY.clear()
            for (v: Double <- value.asInstanceOf[java.util.List[Double]].asScala) batchY append v
          } catch {
            case e: Exception =>
              println("Error while trying to update the batchY hyper parameter of the Neural Network classifier.")
              e.printStackTrace()
          }
        case _ =>
      }
    }
    this
  }

  override def setParametersFromMap(hyperParameterMap: mutable.Map[String, AnyRef]): Learner = {
    for ((parameter, value) <- hyperParameterMap) {
      parameter match {
        case "parameters" =>
          if (parametersSize != null && parametersSize.nonEmpty) {
            try {
              setParams(Nd4j.create(value.asInstanceOf[java.util.List[Double]].asScala.toArray, parametersSize, 'c'))
            } catch {
              case e: Exception =>
                println("Error while trying to update the learnable parameters of the Neural Network classifier.")
                e.printStackTrace()
            }
          }
        case _ =>
      }
    }
    this
  }

  override def setStructureFromMap(structureMap: mutable.Map[String, AnyRef]): Learner = {
    for ((parameter, value) <- structureMap) {
      parameter match {
        case "config" =>
          if (inputShape != null && inputShape.nonEmpty && numOfClasses != 0) {
            try {
              confFromJson(value.asInstanceOf[String], inputShape, numOfClasses)
            } catch {
              case e: Exception =>
                println("Error while trying to update the configuration of the Neural Network classifier.")
                e.printStackTrace()
            }
          }
        case _ =>
      }
    }
    this
  }

  override def generateParameters: ParameterDescriptor => LearningParameters = weights.generateParameters

  override def extractParams: (LearningParameters, Boolean) => SerializableParameters = weights.extractParams

  override def extractDivParams: (LearningParameters , Array[_]) => Array[Array[SerializableParameters]] =
    weights.extractDivParams

  override def generatePOJOLearner: LearnerPOJO = {
    new LearnerPOJO("NN",
      Map[String, AnyRef](
        ("inputShape", inputShape.asInstanceOf[AnyRef]),
        ("numOfClasses", numOfClasses.asInstanceOf[AnyRef]),
        ("parametersSize", getParams.shape().map(_.toInt).asInstanceOf[AnyRef]),
        ("miniBatchSize", getMiniBatchSize.asInstanceOf[AnyRef]),
        ("batchX", batchX.toList.asJava.asInstanceOf[AnyRef]),
        ("batchY", batchY.toList.asJava.asInstanceOf[AnyRef])
      ).asJava,
      Map[String, AnyRef](("parameters", getParams.toDoubleVector.asInstanceOf[AnyRef])).asJava,
      Map[String, AnyRef](("config", jsonConf.asInstanceOf[AnyRef])).asJava
    )
  }

}

object NeuralNetwork {

  def apply(): NeuralNetwork = {
    val default = defaultCNN()
    NeuralNetwork(default._1, default._2, default._3)
  }

  def defaultCNN(): (MultiLayerConfiguration, Array[Int], Int) = {
    val channels = 1
    val n_classes = 10
    val conf = new NeuralNetConfiguration.Builder()
      .seed(1234)
      .l2(0.0005)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new Adam())
      .weightInit(WeightInit.XAVIER)
      .list.layer(new ConvolutionLayer.Builder(5, 5)
      .nIn(channels)
      .stride(2, 2)
      .convolutionMode(ConvolutionMode.Same)
      .nOut(32)
      .activation(Activation.IDENTITY).build)
      .layer(new BatchNormalization.Builder().build)
      .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
      .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2).build)
      .layer(new ConvolutionLayer.Builder(5, 5)
        .stride(2, 2)
        .convolutionMode(ConvolutionMode.Same)
        .nOut(64)
        .activation(Activation.IDENTITY).build)
      .layer(new BatchNormalization.Builder().build)
      .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
      .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2).build)
      .layer(new DenseLayer.Builder()
        .nOut(256)
        .activation(Activation.IDENTITY).build)
      .layer(new BatchNormalization.Builder().build)
      .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
      .layer(new DropoutLayer())
      .layer(new DenseLayer.Builder()
        .activation(Activation.RELU)
        .nOut(10).build)
      .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(n_classes)
        .activation(Activation.SOFTMAX).build)
      .setInputType(InputType.convolutionalFlat(28, 28, 1))
      .build // InputType.convolutional for normal image
    (conf, Array(1, 784), n_classes)
  }

}