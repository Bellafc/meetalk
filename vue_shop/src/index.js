import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    isLoggedIn: false // 默认未登录
  },
  mutations: {
    // 更新登录状态为已登录
    SET_LOGIN_STATE(state, isLoggedIn) {
      state.isLoggedIn = isLoggedIn
    }
  },
  actions: {
    // 登录操作
    login({ commit }) {
      // 在这里可以执行登录逻辑，比如验证用户信息等
      // 假设登录成功
      commit('SET_LOGIN_STATE', true)
    },
    // 退出登录操作
    logout({ commit }) {
      // 在这里可以执行退出登录逻辑，比如清除用户信息等
      // 假设退出成功
      commit('SET_LOGIN_STATE', false)
    }
  }
})
