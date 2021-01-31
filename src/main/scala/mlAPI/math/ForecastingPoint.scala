package mlAPI.math

case class ForecastingPoint(var forecastingPoint: UnlabeledPoint) extends UsablePoint {

  require(forecastingPoint.isInstanceOf[UnlabeledPoint])

  def this() = this(new UnlabeledPoint())

  def getForecastingPoint: UnlabeledPoint = forecastingPoint

  def setForecastingPoint(forecastingPoint: UnlabeledPoint): Unit = this.forecastingPoint = forecastingPoint

  override def asTrainingPoint: TrainingPoint = TrainingPoint(forecastingPoint)

  override var numericVector: Vector = forecastingPoint.numericVector

  override var discreteVector: Vector = forecastingPoint.discreteVector

  override var categoricalVector: Array[String] = forecastingPoint.categoricalVector

  override var dataInstance: String = forecastingPoint.dataInstance

}