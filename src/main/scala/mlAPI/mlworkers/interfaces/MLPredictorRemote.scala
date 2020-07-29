package mlAPI.mlworkers.interfaces

import BipartiteTopologyAPI.annotations.RemoteOp
import ControlAPI.Prediction

trait MLPredictorRemote {
  @RemoteOp
  def sendPrediction(prediction: Prediction): Unit
}
