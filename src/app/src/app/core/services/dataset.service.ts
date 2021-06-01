import {Injectable} from "@angular/core";
import {HttpClient} from "@angular/common/http";
import {Observable} from "rxjs";
import {Dataset} from "../models/dataset";


@Injectable()
export class DatasetService {

  apiUrl = 'http://localhost:4200/api/dataset';

  constructor(private http: HttpClient) {
  }

  postDataset(dataset: Dataset): Observable<Dataset> {
    return this.http.post<Dataset>(this.apiUrl + "/create", dataset);
  }

  updateDataset(dataset: Dataset): Observable<Dataset> {
    return this.http.put<Dataset>(this.apiUrl + '/' + dataset.id, dataset);
  }

  getDatasets(): Observable<Dataset[]> {
    return this.http.get<Dataset[]>(`${this.apiUrl}`);
  }
}
