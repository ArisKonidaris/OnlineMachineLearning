package mlAPI.parameters

import mlAPI.math.{DenseVector, SparseVector, Vector}

trait VectoredParameters extends LearningParameters {

  def getSizes: Array[Int]

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

  def FrobeniusNorm: Double

  def toDenseVector: Vector

  def toSparseVector: Vector

  def slice(range: Bucket, sparse: Boolean): Vector

  def slice(range: Bucket): Vector = slice(range, sparse = false)

  def toDense(vector: Vector): DenseVector = {
    vector match {
      case dense: DenseVector => dense
      case sparse: SparseVector => sparse.toDenseVector
      case _ => throw new RuntimeException("Unknown vector type.")
    }
  }

}
