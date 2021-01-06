package mlAPI.preprocessing

case class RunningMean() extends Serializable {

  private var mean: Double = _
  private var count: Long = _

  def init(value: Double): Unit = {
    mean = value
    count = 0L
  }

  def update(value: Double): Unit = {
    try {
      count += 1
      mean = mean + (1 / (1.0 * count)) * (value - mean)
    } catch {
      case _: Throwable =>
        init(value)
        update(value)
    }
  }

  def setMean(mean: Double): RunningMean = {
    this.mean = mean
    this
  }

  def getMean: Double = {
    val value = mean
    value
  }

  def setCount(count: Long): RunningMean = {
    this.count = count
    this
  }

  def getCount: Long = {
    val value = count
    value
  }

}
