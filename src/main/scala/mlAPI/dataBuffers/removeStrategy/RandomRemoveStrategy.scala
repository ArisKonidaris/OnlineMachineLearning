package mlAPI.dataBuffers.removeStrategy

import mlAPI.dataBuffers.DataSet

import scala.util.Random

case class RandomRemoveStrategy[T]() extends RemoveStrategy[T] {
  override def removeTuple(dataSet: DataSet[T]): Option[T] = dataSet.remove(Random.nextInt(dataSet.length))
}

