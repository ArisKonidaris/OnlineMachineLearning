package mlAPI.math

import breeze.linalg.{Vector => BreezeVector}

/** Type class which allows the conversion from Breeze vectors to Flink vectors
  *
  * @tparam T Resulting type of the conversion, subtype of [[Vector]]
  */
trait BreezeVectorConverter[T <: Vector] extends Serializable {
  /** Converts a Breeze vector into a Flink vector of type T
    *
    * @param vector Breeze vector
    * @return Flink vector of type T
    */
  def convert(vector: BreezeVector[Double]): T
}
