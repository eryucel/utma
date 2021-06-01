import {Injectable} from "@angular/core";
import {HttpClient} from "@angular/common/http";
import {interval, Observable} from "rxjs";
import {Task} from "../models/task";
import {flatMap} from "rxjs/internal/operators";


@Injectable()
export class TaskService {

  apiUrl = 'http://localhost:4200/api/task';

  constructor(private http: HttpClient) {
  }

  postTask(task: Task): Observable<Task> {
    return this.http.post<Task>(this.apiUrl + "/create", task);
  }

  getTasksInterval(): Observable<Task[]> {
    return interval(3000).pipe(flatMap(() => {
      return this.http.get<Task[]>(`${this.apiUrl}`)
    }));
  }

  getTasks(): Observable<Task[]> {
    return this.http.get<Task[]>(`${this.apiUrl}`);
  }

  runTask(task_id: number | undefined) {
    this.http.post(this.apiUrl + '/run/' + task_id, {}).subscribe();
  }
}
