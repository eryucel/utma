import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';
import {HTTP_INTERCEPTORS} from "@angular/common/http";
import {HttpTokenInterceptor} from "./interceptors";
import {AlertService, ApiService, JwtService, SendDatasetService, UserService} from "./services";
import {AuthGuard} from "./guards/auth-guard.service";
import {DatasetService} from "./services/dataset.service";
import {FileService} from "./services/file.service";
import {TaskService} from "./services/task.service";
import {ResultService} from "./services/result.service";


@NgModule({
  declarations: [],
  imports: [
    CommonModule
  ],
  providers: [
    {provide: HTTP_INTERCEPTORS, useClass: HttpTokenInterceptor, multi: true},
    ApiService,
    AuthGuard,
    JwtService,
    UserService,
    AlertService,
    SendDatasetService,
    FileService,
    DatasetService,
    TaskService,
    ResultService
  ],
})
export class CoreModule {
}
