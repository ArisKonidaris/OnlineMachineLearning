package mlAPI.math

case class TrainingPoint(var trainingPoint: LearningPoint) extends UsablePoint {

  def this() = this(new UnlabeledPoint())

  def getTrainingPoint: LearningPoint = trainingPoint

  def setTrainingPoint(trainingPoint: LearningPoint): Unit = this.trainingPoint = trainingPoint

  override def asForecastingPoint: ForecastingPoint = ForecastingPoint(trainingPoint.asUnlabeledPoint)

  override var numericVector: Vector = trainingPoint.numericVector

  override var discreteVector: Vector = trainingPoint.discreteVector

  override var categoricalVector: Array[String] = trainingPoint.categoricalVector

  override var dataInstance: String = trainingPoint.dataInstance

  override def asTrainingPoint: TrainingPoint = this
}
