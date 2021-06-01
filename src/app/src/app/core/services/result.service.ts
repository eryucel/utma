import {Injectable} from "@angular/core";
import {HttpClient} from "@angular/common/http";
import {Observable} from "rxjs";
import {Result} from "../models/result";


@Injectable()
export class ResultService {

  apiUrl = 'http://localhost:4200/api/result';

  constructor(private http: HttpClient) {
  }

  getResults(): Observable<Result[]> {
    return this.http.get<Result[]>(`${this.apiUrl}`);
  }

  getResult(id: string | null): Observable<Result[]> {
    return this.http.get<Result[]>(`${this.apiUrl}/${id}`);
  }
}
