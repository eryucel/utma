import {Injectable} from '@angular/core';

declare let alertify: any;

@Injectable()
export class AlertService {

  success(message: string): void {
    alertify.success(message);
  }

  error(message: string): void {
    alertify.error(message);
  }

  warning(message: string): void {
    alertify.warning(message);
  }

  message(message: string): void {
    alertify.message(message);
  }

  clear(): void {
    alertify.closeAll();
  }
}
