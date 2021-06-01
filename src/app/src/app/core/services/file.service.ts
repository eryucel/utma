import {Injectable} from "@angular/core";
import {HttpClient, HttpHeaders} from "@angular/common/http";
import {Observable} from "rxjs";


@Injectable()
export class FileService {

  apiUrl = 'http://localhost:4200/api/upload';
  fileUrl = 'http://localhost:4200/files/';

  constructor(private http: HttpClient) {
  }

  postFile(name: string, file: any): Observable<any> {
    let formData: FormData = new FormData();
    formData.append('file', file, name.replace(' ', '_') + '.csv');
    let headers = new HttpHeaders();
    headers.append('Content-Type', 'multipart/form-data');
    headers.append('Accept', 'application/json');
    let options = {headers: headers};
    return this.http.post(this.apiUrl + '/', formData, options);
  }

  getFileUrl(name: string): string {
    return this.fileUrl + name;
  }
}
