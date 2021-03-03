package mlAPI.learners.classification.nn

import org.deeplearning4j.nn.api.Model
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.shape.LongShapeDescriptor

/**
 * A helper class holding information about the parameters of a neural network.
 */
class NNParametersInfo(rank: Int,
                       rows: Int,
                       columns: Int,
                       shape: Array[Int],
                       size: Long,
                       descriptor: LongShapeDescriptor) extends Serializable {

  override def toString: String = {
    "\nParametersInfo:\n" +
      "rank: " + rank + "\n" +
      "rows: " + rows + "\n" +
      "columns: " + columns + "\n" +
      "shape: " + shape.mkString("Array(", ", ", ")") + "\n" +
      "size: " + size + "\n" +
      "shapeDescriptor: " + descriptor
  }

}

object NNParametersInfo {
  def apply(nn: Model): NNParametersInfo = {
    val params: INDArray = nn.params()
    val rank: Int = params.rank()
    val rows: Int = params.rows()
    val columns: Int = params.columns()
    val shape: Array[Int] = params.shape().map(_.toInt)
    val size: Long = params.length()
    val descriptor: LongShapeDescriptor = params.shapeDescriptor()
    new NNParametersInfo(rank, rows, columns, shape, size, descriptor)
  }
}
