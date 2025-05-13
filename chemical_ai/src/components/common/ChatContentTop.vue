<template>
  <div class="introduction">
    <div class="introduction-content">
      <img src="@/assets/product.png" alt="" class="introduction-img">
      <div class="introduction-title">我是天工AI智能助手，很高兴见到你！</div>
      <h3>
        我是安全专家精准问答助手👋，专注于为企业安全管理人员及团队提供精准、高效的安全问题解决方案。我融合了先进的人工智能技术与丰富的安全管理知识库，旨在帮助企业在日常运营中识别、评估企业合规性，确保生产安全、员工健康及企业资产安全。
      </h3>
    </div>
  </div>
</template>

<script>
export default {
  name: "SecurityExpertIntro",
};
</script>

<style scoped>
.introduction {
  margin: 40px auto;
  padding: 20px;
  max-width: 100%;
  border-radius: 15px;
  display: flex;
  justify-content: center;
}

.introduction-content {
  align-items: center;
  display: flex;
  flex-direction: column;
  text-align: center;
  gap: 20px;
  padding: 20px;
}

.introduction-img {
  width: 80px;
  height: 80px;
  object-fit: cover;
}

.introduction-title {
  font-size: 24px;
  font-weight: bold;
  color: #333;
}
</style>