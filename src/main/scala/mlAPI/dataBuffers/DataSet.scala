package mlAPI.dataBuffers

import BipartiteTopologyAPI.interfaces.Mergeable
import mlAPI.dataBuffers.removeStrategy.{RandomRemoveStrategy, RemoveOldestStrategy, RemoveStrategy}

import scala.collection.mutable.ListBuffer

case class DataSet[T](var dataBuffer: ListBuffer[T], var max_size: Int)
  extends DataBuffer[T] {

  def this() = this(ListBuffer[T](), 500000)

  def this(training_set: ListBuffer[T]) = this(training_set, 500000)

  def this(max_size: Int) = this(ListBuffer[T](), max_size)

  /** This is the removal strategy of data from the buffer. */
  var remove_strategy: RemoveStrategy[T] = RemoveOldestStrategy[T]()

  /** This is the removal strategy of data from the buffer when merging two data buffers. */
  var merging_remove_strategy: RemoveStrategy[T] = RandomRemoveStrategy[T]()

  override def isEmpty: Boolean = dataBuffer.isEmpty

  override def append(data: T): Option[T] = {
    dataBuffer += data
    overflowCheck()
  }

  override def insert(index: Int, data: T): Option[T] = {
    dataBuffer.insert(index, data)
    overflowCheck()
  }

  override def length: Int = dataBuffer.length

  override def clear(): Unit = {
    dataBuffer.clear()
    max_size = 500000
  }

  override def pop: Option[T] = remove(0)

  override def remove(index: Int): Option[T] = {
    if (dataBuffer.length > index) Some(dataBuffer.remove(index)) else None
  }

  override def merge(mergeables: Array[Mergeable]): Unit = {
    require(mergeables.isInstanceOf[Array[DataSet[T]]])
    var merges: Int = 0
    for (buffer: DataSet[T] <- mergeables.asInstanceOf[Array[DataSet[T]]]) {
      if (buffer.nonEmpty) {
        if (isEmpty) {
          dataBuffer = buffer.getDataBuffer
        } else {
          merges += 1
          dataBuffer = DatasetUtils.mergeBufferedPoints(1, length,
            0, buffer.length,
            dataBuffer, buffer.getDataBuffer,
            merges)
        }
      }
    }
  }

  def overflowCheck(): Option[T] = {
    if (dataBuffer.length > max_size)
      Some(remove_strategy.removeTuple(this).get)
    else
      None
  }

  /** A method that signals the end of the merging procedure of DataBuffer objects. */
  def completeMerge(): Option[ListBuffer[T]] =
    if (length > max_size) Some(merging_remove_strategy.remove(this)) else None

  /////////////////////////////////////////// Getters ////////////////////////////////////////////////

  def getDataBuffer: ListBuffer[T] = dataBuffer

  def getMaxSize: Int = max_size

  def getRemoveStrategy: RemoveStrategy[T] = remove_strategy

  def getMergingRemoveStrategy: RemoveStrategy[T] = merging_remove_strategy

  /////////////////////////////////////////// Setters ////////////////////////////////////////////////

  def setDataBuffer(data_set: ListBuffer[T]): Unit = this.dataBuffer = data_set

  def setMaxSize(max_size: Int): Unit = this.max_size = max_size

  def setRemoveStrategy(remove_strategy: RemoveStrategy[T]): Unit = this.remove_strategy = remove_strategy

  def setMergingRemoveStrategy(merging_remove_strategy: RemoveStrategy[T]): Unit =
    this.merging_remove_strategy = merging_remove_strategy

}
