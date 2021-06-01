import {Component, OnInit} from '@angular/core';
import {JwtService, User, UserService} from "../../../core";

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.css']
})
export class HeaderComponent implements OnInit {

  currentUser!: User;

  constructor(private jwtSevice: JwtService,
              private userService: UserService) {
  }

  logout() {
    this.jwtSevice.destroyToken();
  }

  ngOnInit(): void {
    this.userService.currentUser.subscribe(
      (userData) => {
        this.currentUser = userData;
      }
    );
  }

}
