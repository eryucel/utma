import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';
import {HTTP_INTERCEPTORS} from "@angular/common/http";
import {HttpTokenInterceptor} from "./interceptors";
import {AlertService, ApiService, JwtService, SendDatasetService, UserService} from "./services";
import {AuthGuard} from "./guards/auth-guard.service";


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
    SendDatasetService
  ],
})
export class CoreModule {
}
