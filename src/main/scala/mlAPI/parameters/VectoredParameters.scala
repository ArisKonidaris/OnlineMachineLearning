package mlAPI.parameters

import mlAPI.math.{DenseVector, SparseVector, Vector}
import mlAPI.parameters.utils.Bucket

trait VectoredParameters extends LearningParameters {

  def +(num: Double): LearningParameters

  def +=(num: Double): LearningParameters

  def +(params: LearningParameters): LearningParameters

  def +=(params: LearningParameters): LearningParameters

  def -(num: Double): LearningParameters

  def -=(num: Double): LearningParameters

  def -(params: LearningParameters): LearningParameters

  def -=(params: LearningParameters): LearningParameters

  def *(num: Double): LearningParameters

  def *=(num: Double): LearningParameters

  def /(num: Double): LearningParameters

  def /=(num: Double): LearningParameters

  def frobeniusNorm: Double

  def toDenseVector: Vector

  def toSparseVector: Vector

  def slice(range: Bucket, sparse: Boolean = false): Vector

  def toDense(vector: Vector): DenseVector = {
    vector match {
      case dense: DenseVector => dense
      case sparse: SparseVector => sparse.toDenseVector
      case _ => throw new RuntimeException("Unknown vector type.")
    }
  }

  def extractSparseQuantiles(args: Array[_]): (Boolean, Array[Array[Bucket]]) =
    (args.head.asInstanceOf[Boolean], args.tail.head.asInstanceOf[Array[Array[Bucket]]])

}
