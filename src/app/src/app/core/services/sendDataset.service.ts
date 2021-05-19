import {Injectable} from '@angular/core';
import {UploadedDataset} from "../models/uploadedDataset";

@Injectable()
export class SendDatasetService {
  datasetDetail: UploadedDataset = new UploadedDataset();

  setDatasetDetail(dataset: UploadedDataset) {
    this.datasetDetail = dataset;
  }

  constructor() {
  }
}
